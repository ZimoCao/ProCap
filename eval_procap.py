import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
from collections import OrderedDict
from common.dist_utils import (
    init_distributed_mode,
    get_rank,
    get_world_size,
    is_main_process,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from models.procap import ProCap
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_image(img_path):
    if isinstance(img_path, str):
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return transform(img)  # Return tensor directly, unsqueeze in dataset

class EvalDataset(Dataset):
    def __init__(self, annotations, image_folder, task, preprocess_fn):
        self.annotations = annotations
        self.image_folder = image_folder
        self.task = task
        self.transform = preprocess_fn

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        image_relative_path = ann["image"]

        image_path = os.path.join(self.image_folder, image_relative_path.split('/')[-2], image_relative_path.split('/')[-1])

        image = self.transform(image_path)

        sample = {
            "image": image,
            "image_id": ann["image_id"],
        }

        if self.task == "scene":
            sample["scene_captions"] = ann["scene_captions"]
            sample["captions"] = ann["scene_captions"]  

        elif self.task == "projection":
            sample["proj_captions"] = ann["projection_captions"]
            sample["captions"] = ann["projection_captions"] 

        elif self.task == "all":
            sample["scene_captions"] = ann["scene_captions"]
            sample["proj_captions"] = ann["projection_captions"]
            sample["captions"] = ann["scene_captions"] 

        return sample

def gather_results(predictions):
    if not args.distributed:
        return predictions

    # Gather predictions from all processes
    all_predictions = [None] * get_world_size()
    torch.distributed.all_gather_object(all_predictions, predictions)

    # Concatenate them on the main process
    if is_main_process():
        gathered_list = []
        for p in all_predictions:
            gathered_list.extend(p)
        return gathered_list
    return None


@torch.no_grad()
def evaluation(args, model):
    device = torch.device(args.device)

    # Load annotations
    annotations = []
    print(
        f"Loading annotations for {args.dataset} from: {args.path_of_val_datasets}"
    )
    if args.dataset in ["coco", "nocaps", "whoops"]:
        with open(args.path_of_val_datasets, "r", encoding="utf-8") as f:
            for line in f:
                annotations.append(json.loads(line.strip()))
    else:
        raise NotImplementedError(
            f"Dataset loading for {args.dataset} not implemented."
        )

    # Setup Dataset, Sampler, and DataLoader
    dataset = EvalDataset(annotations, args.image_folder, args.task, preprocess_image)

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=args.bs,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4,  # You can adjust this
    )

    predicts_scene = []
    predicts_proj = []

    iterable = tqdm(data_loader) if is_main_process() else data_loader

    for samples in iterable:
        images = samples["image"].to(device)
        image_ids = samples["image_id"]
        gt_captions = samples["captions"]

        eval_samples = {"image": images}

        with torch.amp.autocast("cuda", enabled=True):
            # Use model.module to access generate method when using DDP
            model_to_generate = model.module if args.distributed else model
            scene_caps, proj_caps = model_to_generate.generate(
                eval_samples,
                num_beams=args.beam_width,
                max_length=120,
            )

        gt_scene_all = samples.get("scene_captions", samples["captions"])
        gt_proj_all = samples.get("proj_captions", samples["captions"])

        for i in range(len(scene_caps)):
            image_id_item = image_ids[i]
            if isinstance(image_id_item, torch.Tensor):
                image_id_item = image_id_item.item()

            current_scene_gt = [gt_scene_all[j][i] for j in range(len(gt_scene_all))]
            current_proj_gt = [gt_proj_all[j][i] for j in range(len(gt_proj_all))]

            if args.task in ("scene", "all"):
                predicts_scene.append({
                    "image_name": image_id_item,
                    "captions": current_scene_gt,
                    "prediction": scene_caps[i].split('\n')[0].strip(),
                })

            if args.task in ("projection", "all"):
                predicts_proj.append({
                    "image_name": image_id_item,
                    "captions": current_proj_gt,
                    "prediction": proj_caps[i].split('\n')[0].strip(),
                })

    all_scene = gather_results(predicts_scene) if args.task in ("scene", "all") else None
    all_proj  = gather_results(predicts_proj)  if args.task in ("projection", "all") else None

    if is_main_process():
        os.makedirs(args.out_path, exist_ok=True)

        if args.seen_scene:
            flag = "seen"
        elif args.unseen_scene:
            flag = "unseen"
        elif args.newsetting:
            flag = "newsetting"
        else:
            flag = "seen"
            
        if args.task in ("scene", "all"):
            scene_path = os.path.join(
                args.out_path,
                f"{args.dataset}_scene_captions_on_{flag}_scene.json",
            )
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(all_scene, f, indent=4)
            print(f"Saved scene results to {scene_path}")

        if args.task in ("projection", "all"):
            proj_path = os.path.join(
                args.out_path,
                f"{args.dataset}_projection_captions_on_{flag}_scene.json",
            )
            with open(proj_path, "w", encoding="utf-8") as f:
                json.dump(all_proj, f, indent=4)
            print(f"Saved projection results to {proj_path}")


@torch.no_grad()
def main(args):
    if args.distributed:
        init_distributed_mode(args)

    device = torch.device(args.device)

    if not args.disable_random_seed:
        set_seed(args.random_seed)

    model_type = args.model_type

    ckpt = args.ckpt_path
    print("Loading checkpoint from:", ckpt)

    model = ProCap(
        ext_path="ext_data/ext_memory_lvis.pkl",
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
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

    state_dict = torch.load(ckpt, map_location="cpu")["model"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    # --- NEW: Wrap model in DDP ---
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )

    model.eval()

    evaluation(args, model)


if __name__ == "__main__":
    data_root = "/path/to/your/directory"  # <-- PLEASE CHANGE THIS
    
    print("Starts Evaluation...")
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="openlm-research/open_llama_3b", help="Model type for LLM")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", default=True)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--bs", type=int, default=4, help="Batch size PER GPU")
    parser.add_argument("--dataset", default="coco", choices=("coco", "nocaps", "whoops"))
    parser.add_argument("--seen_scene", action="store_true")
    parser.add_argument("--unseen_scene", action="store_true")
    parser.add_argument("--newsetting", action="store_true")
    parser.add_argument("--path_of_val_datasets", default=None)
    parser.add_argument("--image_folder", default=None)
    parser.add_argument("--out_path", default="./generated_captions/")
    parser.add_argument("--ckpt_path", type=str, help="Path to your trained model checkpoint")
    parser.add_argument("--num_query_token_txt", type=int, default=8)
    parser.add_argument("--topn", type=int, default=9)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--task", default="projection", choices=("scene", "projection", "all"))
    parser.add_argument("--disable_random_seed", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=42)
    
    parser.add_argument("--disable_refinement", action="store_true")
    parser.add_argument("--disable_mask", action="store_true")
    parser.add_argument("--disable_scene_qformer", action="store_true")
    parser.add_argument("--disable_proj_qformer", action="store_true")
    
    args = parser.parse_args()
    
    DATASET_CONFIG = {
        "coco": {
            "val": {
                "seen": f"{data_root}/eval_coco_scenes_1_to_60.jsonl",
                "unseen": f"{data_root}/eval_coco_scenes_61_to_65.jsonl",
                "newsetting": f"{data_root}/eval_coco_scenes_66_to_70.jsonl",
            },
            "img": f"{data_root}/eval_coco",
        },
        "nocaps": {
            "val": {
                "seen": f"{data_root}/eval_nocaps_scenes_1_to_60.jsonl",
                "unseen": f"{data_root}/eval_nocaps_scenes_61_to_65.jsonl",
                "newsetting": f"{data_root}/eval_nocaps_scenes_66_to_70.jsonl",
            },
            "img": f"{data_root}/eval_nocaps",
        },
        "whoops": {
            "val": {
                "seen": f"{data_root}/eval_whoops_scenes_1_to_60.jsonl",
                "unseen": f"{data_root}/eval_whoops_scenes_61_to_65.jsonl",
                "newsetting": f"{data_root}/eval_whoops_scenes_66_to_70.jsonl",
            },
            "img": f"{data_root}/eval_whoops",
        },
    }
    cfg = DATASET_CONFIG[args.dataset]

    if args.seen_scene:
        scene = "seen"
    elif args.unseen_scene:
        scene = "unseen"
    elif args.newsetting:
        scene = "newsetting"
    else:
        scene = "seen"
    if args.path_of_val_datasets is None:
        args.path_of_val_datasets = cfg["val"][scene]
    if args.image_folder is None:
        args.image_folder = cfg["img"]

    print("Evaluation args: {}\n".format(vars(args)))
    main(args)
