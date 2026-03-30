import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from models.blip2 import Blip2Base, disabled_train
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import pickle
import faiss

transformers.logging.set_verbosity_error()

class ProCap(Blip2Base):

    def __init__(
        self,
        ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llm_model="",
        max_txt_len=160,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        with_refinement=True,
        with_mask=True,
        with_scene_qfromer=True,
        with_proj_qformer=True
    ):
        super().__init__()

        self.low_resource = low_resource
        self.llm_model_name = llm_model
        self.topn = topn

        self.with_refinement = with_refinement
        self.with_mask = with_mask
        self.with_scene_qfromer = with_scene_qfromer
        self.with_proj_qformer = with_proj_qformer
        
        # Visual Backbone (ViT)
        print("Loading ViT")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")

        self.feature_dim = 256 if self.with_refinement else self.visual_encoder.num_features

        # Specialized Q-Formers
        print("Loading Scene Q-Former")
        
        self.qformer_scene, self.query_tokens_scene = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.qformer_scene.cls = None
        self.qformer_scene.bert.embeddings.word_embeddings = None
        self.qformer_scene.bert.embeddings.position_embeddings = None
        for layer in self.qformer_scene.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        
        if freeze_qformer:
            for name, param in self.qformer_scene.named_parameters():
                trainable_parts = ['crossattention']
                is_trainable = any(part in name for part in trainable_parts)
                if not is_trainable:
                    param.requires_grad = False

            self.qformer_scene = self.qformer_scene.eval()
            self.qformer_scene.train = disabled_train
            self.query_tokens_scene.requires_grad = True

        print("Loading Scene Q-Former Done")

        print("Loading Projection Q-Former")
        
        self.qformer_proj, self.query_tokens_proj = self.init_Qformer(
            num_query_token, self.feature_dim
        )
        self.qformer_proj.cls = None
        self.qformer_proj.bert.embeddings.word_embeddings = None
        self.qformer_proj.bert.embeddings.position_embeddings = None
        for layer in self.qformer_proj.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        
        if freeze_qformer:
            for name, param in self.qformer_proj.named_parameters():
                trainable_parts = ['crossattention']
                is_trainable = any(part in name for part in trainable_parts)
                if not is_trainable:
                    param.requires_grad = False
            self.qformer_proj = self.qformer_proj.eval()
            self.qformer_proj.train = disabled_train
            self.query_tokens_proj.requires_grad = True

        print("Loading Projection Q-Former Done")

        print("Loading Knowledge Q-Former for Retrieval")
        
        self.bert_tokenizer = self.init_tokenizer()
        self.qformer_kn, self.query_tokens_kn = self.init_Qformer_kn(
            num_query_token_txt, self.qformer_proj.config.hidden_size
        )
        self.qformer_kn.resize_token_embeddings(len(self.bert_tokenizer))
        self.qformer_kn.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.qformer_kn.named_parameters():
                trainable_parts = ['crossattention']
                is_trainable = any(part in name for part in trainable_parts)
                if not is_trainable:
                    param.requires_grad = False
            self.qformer_kn = self.qformer_kn.eval()
            self.qformer_kn.train = disabled_train
            self.query_tokens_kn.requires_grad = True

        print("Loading Knowledge Q-Former Done")
    
        # Feature Extraction and Projection Segmentation
        if self.with_refinement:
            print("Loading Feature Refinement Module")
            self.feature_refinement = nn.Sequential(
                nn.ConvTranspose2d(
                    self.visual_encoder.num_features, 768, kernel_size=2, stride=2
                ),
                nn.GELU(),
                nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            )
        else:
            self.feature_refinement = None

        print("Loading Projection Segmentation")
        self.proj_segment_model = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        if not self.with_mask:
            for p in self.proj_segment_model.parameters():
                p.requires_grad = False

        # Large Language Model (LLM)
        print("Loading LLM")
        
        config = AutoConfig.from_pretrained(self.llm_model_name)

        # Tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        if self.low_resource:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                config=config,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                config=config,
                torch_dtype=torch.float16,
            )

        # Freeze Parameters
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        print("Loading LLM Done")

        # Connectors and Task Tokens
        # Projects Q-Former outputs into LLM's embedding space
        self.connector = nn.Linear(
            self.qformer_scene.config.hidden_size, self.llm_model.config.hidden_size
        )

        # Learnable tokens to instruct the LLM
        self.scene_task_token = nn.Parameter(
            torch.zeros(1, 1, self.llm_model.config.hidden_size)
        )
        self.projection_task_token = nn.Parameter(
            torch.zeros(1, 1, self.llm_model.config.hidden_size)
        )
        nn.init.normal_(self.scene_task_token, std=0.02)
        nn.init.normal_(self.projection_task_token, std=0.02)

        # Disable Scene branch
        if not self.with_scene_qfromer:
            for p in self.qformer_scene.parameters():
                p.requires_grad = False
            self.query_tokens_scene.requires_grad = False
            self.scene_task_token.requires_grad = False
            
        # Disable Projection branch
        if not self.with_proj_qformer:
            for p in self.qformer_proj.parameters():
                p.requires_grad = False
            self.query_tokens_proj.requires_grad = False
            self.projection_task_token.requires_grad = False
            # Knowledge Q-Former is only used in the Projection branch
            for p in self.qformer_kn.parameters():
                p.requires_grad = False
            self.query_tokens_kn.requires_grad = False

        # External Knowledge Base (FAISS)
        print(f"Loading external knowledge base from: {ext_path}")
        with open(ext_path, "rb") as f:
            ext_base_img, self.ext_base_img_id = pickle.load(f)
            feature_library_cpu = ext_base_img.cpu().numpy()
            faiss.normalize_L2(feature_library_cpu)
            self.feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
            self.feat_index.add(feature_library_cpu)
        print("External knowledge base loaded.")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_multitask(self, image):
        device = image.device

        autocast_context = self.maybe_autocast()

        with autocast_context:
            # Get high-resolution features
            # Note: ViT-G output is (B, 257, 1408). We drop CLS token and reshape.
            low_res_features_flat = self.ln_vision(self.visual_encoder(image))[:, 1:, :]
            low_res_features = low_res_features_flat.transpose(1, 2).reshape(
                -1, 1408, 16, 16
            )
            if self.with_refinement:
                high_res_features = self.feature_refinement(low_res_features)
            else:
                high_res_features = low_res_features

            # Predict Mask
            predicted_mask = None
            masked_features = high_res_features

            if self.with_mask:
                predicted_mask = self.proj_segment_model(high_res_features)
                # Apply Mask Pooling
                masked_features = high_res_features * predicted_mask
            else:
                masked_features = high_res_features
                predicted_mask = None
            
            if self.with_proj_qformer:
                masked_features_flat = masked_features.flatten(2).transpose(1, 2)

                proj_query_tokens = self.query_tokens_proj.expand(image.shape[0], -1, -1)
                proj_query_output = self.qformer_proj(
                    query_embeds=proj_query_tokens,
                    encoder_hidden_states=masked_features_flat,
                    return_dict=True,
                ).last_hidden_state

                # Retrieve knowledge
                re_txt_list_batch = self.retrieve_similar_features(proj_query_output)
                text = self.bert_tokenizer(
                    re_txt_list_batch,
                    truncation=True,
                    padding="longest",
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(device)

                # Get text features
                query_tokens_kn = self.query_tokens_kn.expand(image.shape[0], -1, -1)
                
                proj_atts = torch.ones(proj_query_output.size()[:-1], dtype=torch.long).to(device)
                query_atts_txt = torch.ones(query_tokens_kn.size()[:-1], dtype=torch.long).to(device)
                self_attention_mask = torch.cat([query_atts_txt, text.attention_mask], dim=1)
                
                text_query_output = self.qformer_kn(
                    input_ids=text.input_ids,
                    query_embeds=query_tokens_kn,
                    attention_mask=self_attention_mask,
                    encoder_hidden_states=proj_query_output,
                    encoder_attention_mask=proj_atts, 
                    return_dict=True,
                ).last_hidden_state

                # Combine
                proj_combined_query = torch.cat(
                    [proj_query_output, text_query_output], dim=1
                )
                final_proj_embedding = self.connector(proj_combined_query)
            else:
                final_proj_embedding = masked_features.sum(dim=1, keepdim=True) * 0.0
               
            if self.with_scene_qfromer:
                scene_query_tokens = self.query_tokens_scene.expand(image.shape[0], -1, -1)
                image_atts = torch.ones(
                    low_res_features_flat.size()[:-1], dtype=torch.long
                ).to(device)
                scene_query_output = self.qformer_scene(
                    query_embeds=scene_query_tokens,
                    encoder_hidden_states=low_res_features_flat,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                ).last_hidden_state

                final_scene_embedding = self.connector(scene_query_output)
            else:
                final_scene_embedding = low_res_features_flat.sum(dim=1, keepdim=True) * 0.0
                
        return predicted_mask, final_scene_embedding, final_proj_embedding

    def forward(self, samples):
        image = samples["image"]
        device = image.device

        # Encode Image and Predict Mask
        predicted_mask, scene_embeds, proj_embeds = self.encode_multitask(image)

        # Calculate Mask Loss
        
        loss_roi = image.sum() * 0.0
        
        if self.with_mask and predicted_mask is not None:
            gt_mask = samples.get("gt_mask")
            if gt_mask is not None:
                if gt_mask.shape[-2:] != predicted_mask.shape[-2:]:
                    gt_mask = F.interpolate(
                        gt_mask,
                        size=predicted_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss_roi = F.binary_cross_entropy_with_logits(predicted_mask, gt_mask)

        # Calculate Captioning Loss
        loss_scene = scene_embeds.sum() * 0.0
        if self.with_scene_qfromer and scene_embeds is not None:
            loss_scene = self._calculate_caption_loss(
                task_token=self.scene_task_token,
                vision_embeds=scene_embeds,
                text_captions=samples["scene_caption"],
            )

        loss_proj = proj_embeds.sum() * 0.0
        if self.with_proj_qformer and proj_embeds is not None:
            loss_proj = self._calculate_caption_loss(
                task_token=self.projection_task_token,
                vision_embeds=proj_embeds,
                text_captions=samples["projection_caption"],
            )

        total_loss = loss_scene + loss_proj + loss_roi

        return {
            "loss": total_loss,
            "loss_scene": loss_scene,
            "loss_proj": loss_proj,
            "loss_roi": loss_roi,
        }

    def _calculate_caption_loss(self, task_token, vision_embeds, text_captions):
        batch_size = vision_embeds.shape[0]

        tokenizer = self.llm_tokenizer
        model = self.llm_model

        # Prepare Text Tokens
        text = [t + self.end_sym for t in text_captions]
        text_tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(vision_embeds.device)

        # Text Embeddings
        text_embeds = model.get_input_embeddings()(text_tokens.input_ids)

        # BOS Token
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=vision_embeds.device) * tokenizer.bos_token_id
        bos_embeds = model.get_input_embeddings()(bos)

        # Task Token
        task_token_embeds = task_token.expand(batch_size, -1, -1)

        # Concatenate Input Embeddings ---
        inputs_embeds = torch.cat([bos_embeds, task_token_embeds, vision_embeds, text_embeds], dim=1)
        
        # Attention Mask
        atts_bos = torch.ones(bos_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device)
        atts_task = torch.ones(task_token_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device)
        atts_vision = torch.ones(vision_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device)
        attention_mask = torch.cat([atts_bos, atts_task, atts_vision, text_tokens.attention_mask], dim=1)

        # Prepare Labels
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == tokenizer.pad_token_id, -100)
        prefix_len = inputs_embeds.shape[1] - text_embeds.shape[1]
        empty_targets = torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=vision_embeds.device)
        targets = torch.cat([empty_targets, targets], dim=1)

        # Forward compute Loss
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True
        )

        return outputs.loss


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=120,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        image = samples["image"]

        _, scene_embeds, proj_embeds = self.encode_multitask(image)
        batch_size = image.shape[0]

        # Generate Scene Caption
        scene_captions = []
        if self.with_scene_qfromer and scene_embeds is not None:
            scene_captions = self._generate_caption_for_task(
                task_token=self.scene_task_token,
                vision_embeds=scene_embeds,
                batch_size=batch_size,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            scene_captions = [""] * batch_size # Placeholder

        # Generate Projection Caption
        proj_captions = []
        if self.with_proj_qformer and proj_embeds is not None:
            proj_captions = self._generate_caption_for_task(
                task_token=self.projection_task_token,
                vision_embeds=proj_embeds,
                batch_size=batch_size,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            proj_captions = [""] * batch_size # Placeholder
        
        return scene_captions, proj_captions

    def _generate_caption_for_task(
        self, task_token, vision_embeds, batch_size, **kwargs
    ):
        device = vision_embeds.device

        bos = (
            torch.ones([batch_size, 1], dtype=torch.long, device=device)
            * self.llm_tokenizer.bos_token_id
        )
        bos_embeds = self.llm_model.get_input_embeddings()(bos)

        task_token_embeds = task_token.expand(batch_size, -1, -1)

        inputs_embeds = torch.cat([bos_embeds, task_token_embeds, vision_embeds], dim=1)
        atts = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(device)

        do_sample = kwargs.pop('use_nucleus_sampling', False)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=atts,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            do_sample=do_sample,
            **kwargs,
        )
        return self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def retrieve_similar_features(self, query_features):
        # Simplified retrieval logic
        dims = query_features.shape[-1]
        query_features_flat = query_features.view(-1, dims)
        query_features_cpu = query_features_flat.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)

        _, top_k_indices = self.feat_index.search(query_features_cpu, self.topn)

        batch_size = query_features.shape[0]
        re_txt_list_batch = []
        for i in range(batch_size):
            # Each item in batch has num_query_token queries. We just use the first one's retrieval results.
            start_index = i * query_features.shape[1]
            indices = top_k_indices[start_index]
            retrieved_captions = [self.ext_base_img_id[idx] for idx in indices]
            # Remove duplicates while preserving order
            unique_captions = list(dict.fromkeys(retrieved_captions))
            re_txt_list_batch.append(" [SEP] ".join(unique_captions))

        return re_txt_list_batch
