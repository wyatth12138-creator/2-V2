from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .cls_head_v2 import PrototypeAwareClsHead
from .dynamic_prompt import DynamicPromptGenerator
from .image_encoder import ImageEncoderViT
from .knowledge_adaptor import KnowledgeAnchorBank
from .mask_decoder import MaskDecoder
from .memory_attention import MemoryAttention
from .memory_bank import MemoryBank
from .memory_encoder import MemoryEncoder
from .prompt_encoder import PromptEncoder
from .task_interaction import PrototypeExtractor, TaskInteractionBlock
from .temporal_observer import CausalTemporalObserver
from .mask_intervention import correct_mask_observations
from utils.prompt_list import PromptBank


class UIVUS(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        model_args=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.model_args = model_args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.enable_dynamic_prompt = not bool(getattr(model_args, "disable_dynamic_prompt", False))
        self.enable_mutual_prompt = not bool(getattr(model_args, "disable_mutual_prompt", False))
        self.enable_prototype_loss = not bool(getattr(model_args, "disable_prototype_loss", False))
        self.enable_memory = not bool(getattr(model_args, "disable_memory", False))
        self.use_recursive_temporal_fusion = not bool(getattr(model_args, "disable_recursive_fusion", False))
        self.memory_retrieval_mode = getattr(model_args, "memory_retrieval_mode", "topk")
        self.memory_topk = int(getattr(model_args, "memory_topk", 10))
        self.interaction_gate_upper = float(getattr(model_args, "interaction_gate_upper", 0.5))

        d_modality, d_mode, d_type, d_pos, d_task = 2, 4, 2, 12, 4
        self.prompt_bank = PromptBank(d_modality, d_mode, d_type, d_pos, d_task, emb=128)
        self.prototype_extractor = PrototypeExtractor(in_dim=256, proto_dim=128)
        self.dynamic_prompt_generator = DynamicPromptGenerator(
            emb_dim=128,
            enable_dynamic_prompt=self.enable_dynamic_prompt,
            enable_mutual_prompt=self.enable_mutual_prompt,
        )
        self.task_interaction = TaskInteractionBlock(
            feat_dim=256,
            proto_dim=128,
            enable_interaction=self.enable_prototype_loss,
            max_gate=self.interaction_gate_upper,
        )
        self.memory_bank = MemoryBank(max_cond=10, max_recent=10, detach=True)
        self.memory_encoder = MemoryEncoder(feat_dim=256, mem_dim=64)
        self.memory_attention = MemoryAttention(feat_dim=256, mem_dim=64, num_heads=8, mlp_ratio=4.0, dropout=0.0)
        self.cls_head = PrototypeAwareClsHead(
            num_classes=1,
            emb=128,
            in_chans=1,
            pretrained_resnet=False,
            pretrained_ckpt=getattr(model_args, "resnet_ckpt", None),
        )
        self.fla_score = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )
        self.temporal_mode = getattr(model_args, "temporal_mode", None) or "baseline"
        if self.temporal_mode not in ("baseline", "fixed", "wavelet", "stft", "adaptive", "projected", "anchored", "registered"):
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")
        self.use_mask_intervention = self.temporal_mode in ("projected", "anchored", "registered")
        observer_mode = "adaptive" if self.use_mask_intervention else self.temporal_mode
        self.temporal_observer = (
            CausalTemporalObserver(133, state_dim=16, mode=observer_mode)
            if self.temporal_mode != "baseline" else None
        )
        self.knowledge_anchor_bank = KnowledgeAnchorBank(feat_dim=128, momentum=0.95)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
            if not any(key in n for key in ("cnn_embed", "post_pos_embed", "Adapter", "rel_pos", "upneck")):
                value.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self, imgs, prompts, pt=None, video_ids=None, mask_valid=None, gt_masks=None,
        use_roi=None, frame_time=None, frame_valid=None, return_temporal=False, **kwargs,
    ):
        imgs5d, (batch_size, time_steps) = self._normalize_imgs(imgs)
        if frame_valid is not None:
            if frame_valid.shape != (batch_size, time_steps):
                raise ValueError("frame_valid must be [B, T]")
            frame_valid = frame_valid.to(device=imgs5d.device, dtype=torch.bool)
            if not frame_valid.any(dim=1).all():
                raise ValueError("Each clip needs at least one valid frame")
            imgs5d = imgs5d.masked_fill(~frame_valid[:, :, None, None, None], 0)
        point_bt = self._normalize_points(pt, B=batch_size, T=time_steps)
        static_prompts = self._repeat_prompts(prompts, time_steps)

        imgs_bt = imgs5d.view(batch_size * time_steps, imgs5d.shape[2], imgs5d.shape[3], imgs5d.shape[4])
        feat_bt = self.image_encoder(imgs_bt, prompts=static_prompts)
        feat_bthw = feat_bt.view(batch_size, time_steps, *feat_bt.shape[1:])

        if time_steps > 1 and self.enable_memory:
            if video_ids is None:
                video_ids = torch.arange(batch_size, device=imgs5d.device, dtype=torch.long)
            self.memory_bank.reset(video_ids)

        low_res_list = []
        mask_list = []
        cls_logits_list = []
        seg_proto_list = []
        cls_proto_list = []
        vid_proto_list = []
        delta_regs = []
        mutual_losses = []
        seg_to_cls_gates = []
        cls_to_seg_gates = []
        memory_gates = []
        prototype_agreements = []
        prototype_confidences = []
        running_cls_proto = None
        observations = []
        quality_scores = []

        for frame_idx in range(time_steps):
            feat_t = feat_bthw[:, frame_idx]
            seed_info = self.prototype_extractor(feat_t)
            seed_proto = seed_info["object_proto"]
            vid_proto = seed_info["task_proto"]
            if time_steps > 1 and self.enable_memory:
                mem_tokens, mem_pad_mask = self._gather_memory_tokens(
                    video_ids,
                    vid_proto,
                    feat_t.device,
                    topk=self.memory_topk,
                )
                if mem_tokens is not None:
                    has_memory = (~mem_pad_mask).any(dim=1)
                    attend = self.memory_attention(
                        feat_t[has_memory], mem_tokens[has_memory],
                        mem_key_padding_mask=mem_pad_mask[has_memory],
                    )
                    feat_t = feat_t.index_copy(0, has_memory.nonzero().flatten(), attend)

            static_prompt_t = self._slice_prompt(static_prompts, batch_size, time_steps, frame_idx)
            dynamic_prompt_t = self.dynamic_prompt_generator(
                static_prompt_t,
                seg_proto=seed_proto,
                cls_proto=running_cls_proto,
                vid_proto=vid_proto,
            )
            prompt_confidence = seed_info["prototype_confidence"].clamp(0.0, 1.0)
            for gate_name in ("seg_to_cls_gate", "cls_to_seg_gate", "memory_gate"):
                dynamic_prompt_t[gate_name] = dynamic_prompt_t[gate_name] * prompt_confidence
            delta_regs.append(dynamic_prompt_t["delta_reg"])
            mutual_losses.append(dynamic_prompt_t["mutual_loss"])
            reliability = {
                "seg_to_cls_gate": dynamic_prompt_t["seg_to_cls_gate"],
                "cls_to_seg_gate": dynamic_prompt_t["cls_to_seg_gate"],
                "memory_gate": dynamic_prompt_t["memory_gate"],
            }
            seg_to_cls_gates.append(dynamic_prompt_t["seg_to_cls_gate"])
            cls_to_seg_gates.append(dynamic_prompt_t["cls_to_seg_gate"])
            memory_gates.append(dynamic_prompt_t["memory_gate"])
            prototype_agreements.append(dynamic_prompt_t["prototype_agreement"])
            prototype_confidences.append(prompt_confidence)

            feat_t, fused_proto = self.task_interaction(
                feat_t,
                seg_proto=dynamic_prompt_t["seg_prompt_proto"],
                cls_proto=dynamic_prompt_t["cls_prompt_proto"],
                vid_proto=dynamic_prompt_t["vid_prompt_proto"],
                reliability=reliability,
            )
            points_t = self._slice_points(point_bt, frame_idx) if point_bt is not None else None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points_t, boxes=None, masks=None)
            low_res_mask_t, _, aux = self.mask_decoder(
                image_embeddings=feat_t,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                return_pointers=True,
            )
            frame_images = imgs5d[:, frame_idx]
            masks_t = F.interpolate(
                low_res_mask_t,
                size=tuple(frame_images.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
            seg_info = self.prototype_extractor(feat_t, mask_logits=low_res_mask_t)
            cls_out = self.cls_head(
                frame_images,
                feat_t,
                mask_logits=low_res_mask_t,
                prompts=dynamic_prompt_t,
                use_roi=use_roi,
                seg_proto=seg_info["task_proto"],
            )
            current_cls_proto = cls_out["cls_proto"].detach()
            if frame_valid is None:
                running_cls_proto = current_cls_proto
            else:
                previous = torch.zeros_like(current_cls_proto) if running_cls_proto is None else running_cls_proto
                running_cls_proto = torch.where(frame_valid[:, frame_idx, None], current_cls_proto, previous)
            if self.temporal_observer is not None and time_steps > 1:
                observation, quality = self._structure_observation(frame_images, low_res_mask_t)
                observations.append(observation)
                quality_scores.append(quality)

            low_res_list.append(low_res_mask_t)
            mask_list.append(masks_t)
            cls_logits_list.append(cls_out["cls_logits"])
            seg_proto_list.append(seg_info["task_proto"])
            cls_proto_list.append(cls_out["cls_proto"])
            vid_proto_list.append(fused_proto)

            if time_steps > 1 and self.enable_memory:
                prompt_mask = self._resolve_prompt_mask(mask_valid, batch_size, time_steps, frame_idx, imgs5d.device)
                memory_mask = torch.sigmoid(low_res_mask_t).detach()
                memory_feat = self.memory_encoder(feat_t, memory_mask)
                if self.use_recursive_temporal_fusion or bool(prompt_mask.any().item()):
                    self.memory_bank.push_spatial(
                        video_ids=video_ids,
                        spatial_bt=memory_feat,
                        pointers_bt=aux["pointers"],
                        frame_idx=frame_idx,
                        prompted_mask=prompt_mask,
                        extra={"proto": vid_proto.detach()},
                        write_mask=None if frame_valid is None else frame_valid[:, frame_idx],
                    )

        outputs = self._format_outputs(
            low_res_list=low_res_list,
            mask_list=mask_list,
            cls_logits_list=cls_logits_list,
            seg_proto_list=seg_proto_list,
            cls_proto_list=cls_proto_list,
            vid_proto_list=vid_proto_list,
            observations=observations, quality_scores=quality_scores,
            frame_time=frame_time, frame_valid=frame_valid, modality=prompts.get("modality_prompt"),
            return_temporal=return_temporal, frames=imgs5d,
        )
        outputs["prompt_reg_loss"] = torch.stack(delta_regs).mean() if delta_regs else imgs5d.new_zeros(())
        outputs["mutual_prompt_loss"] = torch.stack(mutual_losses).mean() if mutual_losses else imgs5d.new_zeros(())
        outputs["seg_to_cls_gate"] = torch.stack(seg_to_cls_gates, dim=1).mean(dim=1) if seg_to_cls_gates else imgs5d.new_zeros((batch_size, 1))
        outputs["cls_to_seg_gate"] = torch.stack(cls_to_seg_gates, dim=1).mean(dim=1) if cls_to_seg_gates else imgs5d.new_zeros((batch_size, 1))
        outputs["memory_gate"] = torch.stack(memory_gates, dim=1).mean(dim=1) if memory_gates else imgs5d.new_zeros((batch_size, 1))
        outputs["prototype_agreement"] = torch.stack(prototype_agreements, dim=1).mean(dim=1) if prototype_agreements else imgs5d.new_zeros((batch_size, 1))
        outputs["prototype_confidence"] = torch.stack(prototype_confidences, dim=1).mean(dim=1) if prototype_confidences else imgs5d.new_zeros((batch_size, 1))
        outputs["anchor_loss"] = imgs5d.new_zeros(())
        outputs["anchor_drift"] = imgs5d.new_zeros(())
        return outputs

    @staticmethod
    def _structure_observation(frame, mask_logits):
        """Display-space ROI signals and a predicted-mask quality proxy."""
        p = mask_logits.detach().float().sigmoid()
        support = p * (p > 0.5)
        mass = support.sum(dim=(2, 3)).clamp_min(1e-6)
        height, width = support.shape[-2:]
        x = torch.linspace(-1, 1, width, device=p.device)
        y = torch.linspace(-1, 1, height, device=p.device)
        area = support.mean(dim=(2, 3))
        center_x = (support * x).sum(dim=(2, 3)) / mass
        center_y = (support * y[:, None]).sum(dim=(2, 3)) / mass
        display = UIVUS._structure_display(frame, (height, width))
        inside = (support * display).sum(dim=(2, 3)) / mass
        outside_weight = 1.0 - support
        outside = (outside_weight * display).sum(dim=(2, 3)) / outside_weight.sum(dim=(2, 3)).clamp_min(1e-6)
        confidence = (support * (2 * p - 1)).sum(dim=(2, 3)) / mass
        # Very small or empty masks provide weak structural side information.
        quality = (confidence * (area / 0.02).clamp(max=1)).squeeze(1)
        return torch.cat((area, center_x, center_y, inside, inside - outside), dim=1), quality

    @staticmethod
    def _structure_display(frame, size):
        display = frame.detach().float() / 255.0
        # Raw enhancement uses channel 1 in the legacy CEUS stack.
        display = display[:, 1:2] if display.shape[1] > 1 else display
        return F.interpolate(display, size=size, mode="area")

    def _format_outputs(
        self, low_res_list, mask_list, cls_logits_list, seg_proto_list, cls_proto_list, vid_proto_list,
        observations, quality_scores, frame_time, frame_valid, modality, return_temporal, frames,
    ):
        if len(low_res_list) == 1:
            return {
                "low_res_logits": low_res_list[0],
                "masks": mask_list[0],
                "cls_logits": cls_logits_list[0],
                "seg_proto": seg_proto_list[0],
                "cls_proto": cls_proto_list[0],
                "vid_proto": vid_proto_list[0],
            }

        frame_cls_proto = torch.stack(cls_proto_list, dim=1)
        temporal = None
        intervention = None
        if self.temporal_observer is not None:
            structure = torch.stack(observations, dim=1).to(frame_cls_proto.dtype)
            quality = torch.stack(quality_scores, dim=1)
            if self.use_mask_intervention:
                masks_low = torch.stack(low_res_list, dim=1)
                height, width = masks_low.shape[-2:]
                display = self._structure_display(frames.flatten(0, 1), (height, width))
                intervention = correct_mask_observations(
                    display.view(*frames.shape[:2], 1, height, width),
                    masks_low.detach().float().sigmoid(), frame_valid,
                    timestamps=frame_time, use_image=self.temporal_mode != "projected",
                    anchor_method="registration" if self.temporal_mode == "registered" else "contour",
                )
                structure = intervention["corrected"].to(frame_cls_proto.dtype)
            modality_ids = modality.argmax(dim=-1) if modality is not None else None
            temporal = self.temporal_observer(
                torch.cat((frame_cls_proto, structure), dim=-1),
                timestamps=frame_time, valid=frame_valid, quality=quality, modality=modality_ids,
            )
            frame_cls_proto = temporal["features"][..., :frame_cls_proto.shape[-1]]

        scores = self.fla_score(frame_cls_proto).squeeze(-1)
        if frame_valid is not None:
            scores = scores.masked_fill(~frame_valid, float("-inf"))
        fla_weights = torch.softmax(scores, dim=1)
        video_cls_proto = torch.sum(fla_weights.unsqueeze(-1) * frame_cls_proto, dim=1)
        output = {
            "low_res_logits": torch.stack(low_res_list, dim=1),
            "masks": torch.stack(mask_list, dim=1),
            "cls_logits": self.cls_head.classifier(video_cls_proto),
            "seg_proto": torch.stack(seg_proto_list, dim=1).mean(dim=1),
            "cls_proto": video_cls_proto,
            "vid_proto": torch.stack(vid_proto_list, dim=1).mean(dim=1),
            "fla_weights": fla_weights,
        }
        if return_temporal and temporal is not None:
            output["temporal"] = {key: value for key, value in temporal.items() if key != "features"}
            output["temporal"]["quality_proxy"] = quality
            if intervention is not None:
                output["temporal"]["mask_intervention"] = intervention
        return output

    def _repeat_prompts(self, prompts: dict, time_steps: int) -> dict:
        base = self.prompt_bank(prompts)
        if time_steps == 1:
            return base
        return {key: value.repeat_interleave(time_steps, dim=0) for key, value in base.items()}

    def _slice_prompt(self, prompts: dict, batch_size: int, time_steps: int, frame_idx: int) -> dict:
        if time_steps == 1:
            return prompts
        sliced = {}
        for key, value in prompts.items():
            sliced[key] = value.view(batch_size, time_steps, -1)[:, frame_idx]
        return sliced

    def _slice_points(self, points_bt: Tuple[torch.Tensor, torch.Tensor], frame_idx: int):
        coords_bt, labels_bt = points_bt
        return coords_bt[:, frame_idx], labels_bt[:, frame_idx]

    def _resolve_prompt_mask(self, mask_valid, batch_size: int, time_steps: int, frame_idx: int, device):
        if mask_valid is None:
            mask = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            mask[0] = True
            return mask
        assert mask_valid.shape[:2] == (batch_size, time_steps)
        return mask_valid[:, frame_idx].to(device=device, dtype=torch.bool)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    def _normalize_imgs(self, imgs):
        if imgs.ndim == 4:
            if imgs.shape[1] in (1, 3) and imgs.shape[0] > 1 and imgs.shape[2] == imgs.shape[3]:
                b, c, h, w = imgs.shape
                return imgs.unsqueeze(1), (b, 1)
            t, c, h, w = imgs.shape
            return imgs.unsqueeze(0), (1, t)
        if imgs.ndim == 5:
            b, t, c, h, w = imgs.shape
            return imgs, (b, t)
        raise ValueError(f"Unexpected imgs.ndim={imgs.ndim}")

    def _normalize_points(self, pt, B, T):
        if pt is None:
            return None
        coords, labels = pt
        entries = coords.shape[0]
        if T == 1:
            return coords.view(B, 1, 1, 2), labels.view(B, 1, 1)
        if B == 1 and entries == T:
            return coords.view(1, T, 1, 2), labels.view(1, T, 1)
        if entries == B * T:
            return coords.view(B, T, 1, 2), labels.view(B, T, 1)
        raise AssertionError(f"Cannot infer pt layout: coords.shape[0]={entries}, B={B}, T={T}")

    def _gather_memory_tokens(self, video_ids, query_proto, device, topk: int = 10):
        batch_size = video_ids.shape[0]
        per_tokens = []
        lengths = []
        for index in range(batch_size):
            vid = int(video_ids[index].item())
            bank = self.memory_bank.get(vid)
            if self.memory_retrieval_mode == "recent_only":
                items = bank["recent"]
            elif self.memory_retrieval_mode == "cond_only":
                items = bank["cond"]
            else:
                items = bank["cond"] + bank["recent"]
            if len(items) == 0:
                per_tokens.append(None)
                lengths.append(0)
                continue

            scored_items = []
            for item in items:
                score = 0.0
                proto = None if item.extra is None else item.extra.get("proto")
                if proto is not None:
                    score = F.cosine_similarity(query_proto[index:index + 1], proto.to(device), dim=-1).item()
                scored_items.append((score, item))
            scored_items.sort(key=lambda entry: entry[0], reverse=True)
            if self.memory_retrieval_mode == "all":
                selected = [item for _, item in scored_items]
            else:
                selected = [item for _, item in scored_items[:topk]]

            token_chunks = []
            for item in selected:
                spatial = item.spatial.to(device).squeeze(0)
                token_chunks.append(spatial.flatten(1).transpose(0, 1))
                if item.pointers is not None:
                    token_chunks.append(item.pointers.to(device).squeeze(0))
            tokens = torch.cat(token_chunks, dim=0)
            per_tokens.append(tokens)
            lengths.append(tokens.shape[0])

        max_len = max(lengths)
        if max_len == 0:
            return None, None

        mem_dim = per_tokens[next(i for i, length in enumerate(lengths) if length > 0)].shape[1]
        mem_tokens = torch.zeros((batch_size, max_len, mem_dim), device=device, dtype=torch.float32)
        key_padding_mask = torch.ones((batch_size, max_len), device=device, dtype=torch.bool)
        for index, tokens in enumerate(per_tokens):
            if tokens is None:
                continue
            mem_tokens[index, : tokens.shape[0]] = tokens
            key_padding_mask[index, : tokens.shape[0]] = False
        return mem_tokens, key_padding_mask
