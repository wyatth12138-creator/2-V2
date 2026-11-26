import torch
import random

def generate_click_prompt(img, msk, pt_label = 1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

# def get_click_prompt(datapack, opt):
#     if 'pt' not in datapack:
#         imgs, pt, masks = generate_click_prompt(imgs, masks)
#     else:
#         pt = datapack['pt']
#         point_labels = datapack['p_label']

#     point_coords = pt
#     coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=opt.device)
#     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=opt.device)
#     if len(pt.shape) == 2:
#         coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#     pt = (coords_torch, labels_torch)
#     return pt

import torch

def _to_1d_labels(x, device):
    """把 0 / [0] / tensor([0]) / [tensor([0])] 等统一成 1D tensor([N])"""
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if torch.is_tensor(x):
        t = x.to(device=device)
    else:
        t = torch.as_tensor(x, device=device)
    return t.reshape(-1)

def get_click_prompt(datapack, opt):
    device = opt.device

    if 'pt' not in datapack:
        # 你原来这里 imgs/masks 未定义，我不改你的结构；
        # 但建议 generate_click_prompt 返回 point_labels
        imgs, pt, masks, point_labels = generate_click_prompt(imgs, masks)
    else:
        pt = datapack['pt']
        point_labels = datapack['p_label']

    # ---------------- coords ----------------
    coords_torch = torch.as_tensor(pt, dtype=torch.float32, device=device)

    # 单图: [N,2] -> [1,N,2]
    if coords_torch.ndim == 2:
        coords_torch = coords_torch.unsqueeze(0)  # [1,N,2]

    # 视频未展平: [B,T,N,2] -> [BT,N,2]
    elif coords_torch.ndim == 4:
        B, T, N, _ = coords_torch.shape
        coords_torch = coords_torch.reshape(B * T, N, 2)

    # 已展平: [BT,N,2] -> 不动
    elif coords_torch.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported pt shape: {tuple(coords_torch.shape)}")

    BT, N, _ = coords_torch.shape

    # ---------------- labels ----------------
    # 目标：labels_torch -> [BT, N]
    if torch.is_tensor(point_labels):
        labels_torch = point_labels.to(device=device)

        # [B,T,N] -> [BT,N]
        if labels_torch.ndim == 3:
            labels_torch = labels_torch.reshape(-1, labels_torch.shape[-1])

        # [T,N] -> 复制到 batch -> [BT,N]
        elif labels_torch.ndim == 2 and labels_torch.shape[0] != BT:
            T = labels_torch.shape[0]
            if BT % T != 0:
                raise ValueError(f"labels [T,N] with T={T} cannot match BT={BT}")
            B = BT // T
            labels_torch = labels_torch.unsqueeze(0).expand(B, -1, -1).reshape(BT, -1)

        # [BT] -> [BT,1]
        elif labels_torch.ndim == 1:
            if labels_torch.numel() != BT and labels_torch.numel() != N:
                raise ValueError(f"1D labels numel={labels_torch.numel()} cannot match BT={BT} or N={N}")
            if labels_torch.numel() == BT:
                labels_torch = labels_torch.view(BT, 1)
            else:  # numel == N (单图常见)
                labels_torch = labels_torch.view(1, N).expand(BT, -1)

        # 兜底把 shape 变成 [BT, ?]
        if labels_torch.ndim != 2 or labels_torch.shape[0] != BT:
            labels_torch = labels_torch.reshape(BT, -1)

    else:
        assert isinstance(point_labels, (list, tuple)), type(point_labels)
        L = len(point_labels)

        # 单图：L==N -> [1,N]
        if BT == 1 and L == N:
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int64, device=device).view(1, N)

        # 已展平：L==BT，每个元素对应一帧(或一张图)的 label
        elif L == BT:
            rows = []
            for i in range(BT):
                rows.append(_to_1d_labels(point_labels[i], device))
            labels_torch = torch.stack(rows, dim=0)  # [BT, ?]

        # 按帧给：L==T，复制到 batch： [T,?] -> [BT,?]
        else:
            T = L
            if BT % T != 0:
                raise ValueError(f"len(p_label)={L} cannot match BT={BT} (BT%T!=0)")
            B = BT // T

            per_frame = []
            for t in range(T):
                per_frame.append(_to_1d_labels(point_labels[t], device))
            labels_t = torch.stack(per_frame, dim=0)  # [T,?]
            labels_torch = labels_t.unsqueeze(0).expand(B, -1, -1).reshape(BT, -1)

    labels_torch = labels_torch.to(dtype=torch.int64, device=device)

    # ---------------- enforce 1-to-1 with points ----------------
    # 必须：每个点一个 label。若你目前生成阶段给多了（例如 [0,0]），这里会截断到 N。
    if labels_torch.shape[1] > N:
        labels_torch = labels_torch[:, :N]
    elif labels_torch.shape[1] < N:
        raise ValueError(f"labels N={labels_torch.shape[1]} < coords N={N}")

    return (coords_torch, labels_torch)

