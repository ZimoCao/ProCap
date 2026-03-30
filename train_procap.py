import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer
from dataset.coco_dataset import COCODataset
from models.procap import ProCap
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, cur_epoch, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
    }
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, output_dir))
    torch.save(save_obj, output_dir)


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[get_rank()], find_unused_parameters=False
        )
    else:
        sampler = None

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    model.train()
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=8e-5,
        init_lr=1e-4,
        decay_rate=None,
        warmup_start_lr=1e-6,
        warmup_steps=5000,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    use_amp = scaler is not None
    print("use_amp", use_amp)

    for epoch in range(epochs):
        if args.distributed and sampler is not None:
            sampler.set_epoch(epoch)
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
        )
        metric_logger.add_meter(
            "loss_scene", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
        )
        metric_logger.add_meter(
            "loss_proj", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
        )
        metric_logger.add_meter(
            "loss_roi", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
        )

        header = "Train Epoch: [{}]".format(epoch)
        for idx, samples in enumerate(
            metric_logger.log_every(train_dataloader, print_freq=50, header=header)
        ):

            samples = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in samples.items()
            }

            scheduler.step(cur_epoch=epoch, cur_step=idx)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(samples)

                loss_scene = loss_dict["loss_scene"]
                loss_proj = loss_dict["loss_proj"]
                loss_roi = loss_dict["loss_roi"]

                # Combine losses with weights from args
                loss = (
                    (args.alpha * loss_scene)
                    + (args.beta * loss_proj)
                    + (args.gamma * loss_roi)
                )

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss_scene=args.alpha * loss_scene.item())
            metric_logger.update(loss_proj=args.beta * loss_proj.item())
            metric_logger.update(loss_roi=args.gamma * loss_roi.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())

        if epoch == epochs - 1:
            if get_rank() == 0:
                output_dir_model = os.path.join(output_dir, f"{epoch:03d}.pt")
                model_to_save = model.module if args.distributed else model
                save_checkpoint(model_to_save, optimizer, epoch, output_dir_model)
    return model


def main():
    data_root = "/path/to/your/directory"  # <-- PLEASE CHANGE THIS
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Starts ...")
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=4)

    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for scene captioning loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for projection captioning loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for projection segmentation loss")

    parser.add_argument("--model_type", type=str, default="openlm-research/open_llama_3b", help="Model type for LLM")
    
    parser.add_argument("--is_rn", dest="is_rn", action="store_true")
    parser.add_argument("--device", default="cuda", help="gpu for training")
    parser.add_argument("--distributed", default=True)
    parser.add_argument("--amp", default=True)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--num_query_token_txt", type=int, default=8)
    parser.add_argument("--topn", type=int, default=9)
    parser.add_argument("--disable_random_seed", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=42)
    
    parser.add_argument("--disable_refinement", action="store_true")
    parser.add_argument("--disable_mask", action="store_true")
    parser.add_argument("--disable_scene_qformer", action="store_true")
    parser.add_argument("--disable_proj_qformer", action="store_true")
    
    args = parser.parse_args()

    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)

    print(f"Training args: {vars(args)}")

    dataset = COCODataset(data_root=data_root)

    model_type = args.model_type

    model = ProCap(
        ext_path="ext_data/ext_memory_lvis.pkl",
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,  # Set to False if you want to finetune Q-Formers
        num_query_token=32,
        num_query_token_txt=args.num_query_token_txt,
        topn=args.topn,
        llm_model=model_type,
        max_txt_len=128,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        with_refinement=not args.disable_refinement,
        with_mask=not args.disable_mask,
        with_scene_qfromer=not args.disable_scene_qformer,
        with_proj_qformer=not args.disable_proj_qformer
    )

    train(dataset, model, args)


if __name__ == "__main__":
    main()
