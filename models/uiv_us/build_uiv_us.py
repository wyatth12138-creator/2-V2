# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer, UIVUS
from torch.nn import functional as F


def build_uiv_us_vit_b(args, checkpoint=None):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    model = UIVUS(
        image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        model_args=args,
    )
    model.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(model, state_dict, image_size, patch_size)
            model.load_state_dict(new_state_dict)
    return model

def load_from(model, pretrained_dict, image_size, patch_size):
    model_dict = model.state_dict()
    dict_trained = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    model_dict.update(dict_trained)
    return model_dict


def load_from2(model, pretrained_dict, image_size, patch_size): # load the positional embedding
    model_dict = model.state_dict()
    dict_trained = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    token_size = int(image_size//patch_size)
    # pos_embed = dict_trained['image_encoder.pos_embed']
    # pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
    # pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
    # pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    # dict_trained['image_encoder.pos_embed'] = pos_embed
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    model_dict.update(dict_trained)
    return model_dict
