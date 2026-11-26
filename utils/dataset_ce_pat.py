import os
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
from random import randint
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from einops import rearrange
import random
from utils.data_us import random_bbox, random_click, fixed_click, fixed_bbox, correct_dims
from utils.split_manifest import (
    assert_disjoint_splits,
    find_dataset_root,
    resolve_partitioned_file,
    split_is_available,
)

# ---------------------- prompt ---------------------- #

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]

from utils.prompt_list import position_prompt_dict

# type prompt
from utils.prompt_list import type_prompt_dict

# organ prompt
from utils.prompt_list import position_prompt_one_hot_dict

# type prompt
from utils.prompt_list import type_prompt_one_hot_dict

# mode prompt
from utils.prompt_list import mode_prompt_one_hot_dict
from utils.prompt_list import modality_prompt_one_hot_dict

# task prompt
from utils.prompt_list import position_prompt_list
from utils.prompt_list import task_prompt_one_hot_dict
from utils.task_registry import TASK_SPECS


CLASS_NAME_TO_LABEL = {
    "0": 0,
    "1": 1,
    "benign": 0,
    "malignant": 1,
    "negative": 0,
    "positive": 1,
    "normal": 0,
    "abnormal": 1,
}


def load_video_and_mask_file(img_path: str,
                             anno_path: str,
                             frame_length: int = 10):
    '''
        modify by ultrasound_npy_sequence_load() from EchoGraphs
        args:
            img_path: videos .npy file path
            anno_path: annotations .npz file path
            frame_length: Placing ED frame at index 0, and ES frame at index -1
        return: 
            imgs: (F,3,112,112)
            masks: (2,112,112)  ED and ES frame
            ef: float
            edv, esv, spacing
    '''
    # load video
    video = np.load(img_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    kpts_list = np.load(anno_path, allow_pickle=True)
    ef, edv, esv = kpts_list['ef'], kpts_list['edv'], kpts_list['esv']

    # Collect masks:
    idx_list = []
    masks = []
    mask_list = kpts_list['fnum_mask'].tolist()
    for kpt in kpts_list['fnum_mask'].tolist().keys():
        idx_list.append(int(kpt))
        masks.append(mask_list[kpt])

    # Swap if ED before ES:
    if idx_list[0] > idx_list[-1]:
        idx_list.reverse()
        masks.reverse()

    # compute step:
    x0, x1 = idx_list[-1], idx_list[0]
    if frame_length == -1:
        frame_length = x0 + 1

    step = min(x0, (x0 - x1) / (frame_length - 1))

    # select frames inds:
    frame_inds = [int(idx_list[0] + step * i) for i in range(frame_length)]

    # Collect frames:
    frames = []
    for i in range(frame_length):
        frames.append(video[frame_inds[i]])

    select_masks = []
    if len(masks) <= len(frame_inds):
        select_masks = masks
    else:
        for i in range(frame_length):
            select_masks.append(masks[frame_inds[i]])

    select_masks = np.asarray(select_masks)
    imgs = np.asarray(frames)

    spacing = kpts_list['spacing']

    return imgs, select_masks, ef, edv, esv, spacing


def _auto_complete_path(root: str, rel: str, default_subdir: Optional[str] = None) -> str:
    """
    root: 数据集根目录，例如 base_dir/segmentation/DATASET_A
    rel:  从 txt 里读出来的一行，例如:
          "imgs/xxx.png"   或 "xxx.png" 或 "xxx"
    default_subdir: 如果 rel 里不包含路径，就加上这个子目录，比如 "imgs"
    """
    rel = rel.strip()
    rel = rel.replace("\\", "/")
    # 如果本身带子目录，直接拼
    if "/" in rel:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    else:
        # 没有子目录，用默认子目录
        if default_subdir is not None:
            path = os.path.join(root, default_subdir, rel).replace("\\", "/")
        else:
            path = os.path.join(root, rel).replace("\\", "/")
        if os.path.exists(path):
            return path

    # 如果没有扩展名，自动尝试常见格式
    base, ext = os.path.splitext(path)
    if ext == "":
        for e in IMG_EXTS:
            p = base + e
            if os.path.exists(p):
                return p

    # 如果有扩展名但文件不存在，换其他扩展名试试
    base, ext = os.path.splitext(path)
    for e in IMG_EXTS:
        p = base + e
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Cannot find file for rel='{rel}' under root='{root}'")


def _parse_seg_config_and_map_mask(
        base_dir: str, dataset_name: str, mask: np.ndarray
) -> np.ndarray:
    """
    按照 omni_dataset 里的 config.yaml 思路，对 mask 做 value->index 映射。
    假设 config.yaml 里每一行大概长这样（你之后可以自定义）:
        0: background: 0
        1: lesion: 255
    这里做一个很宽松的 parser，你可以根据自己实际 config 格式再改。
    """
    import re

    config_path = os.path.join(base_dir, "segmentation", dataset_name, "config.yaml")
    if not os.path.isfile(config_path):
        # 没有 config.yaml 就直接返回原 mask
        return mask

    with open(config_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    # 收集映射: label_value_in_image -> label_index
    mapping: Dict[int, int] = {}
    for line in lines:
        # 用正则找 "index" 和 "value" 两个整数
        # 例如 "0: background: 0" -> index=0, value=0
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= 2:
            label_index = int(nums[0])
            label_value = int(nums[1])
            mapping[label_value] = label_index

    if not mapping:
        return mask

    new_mask = mask.copy()
    for value, idx in mapping.items():
        new_mask[mask == value] = idx
    return new_mask


def _safe_prompt_lookup(mapping: Dict[str, str], dataset_name: str, fallback: str) -> str:
    return mapping.get(dataset_name, fallback)


def _build_task_prompts(dataset_name: str, task_name: str) -> Dict[str, torch.Tensor]:
    prompt_name = TASK_SPECS[task_name].prompt_name
    modality_name = "ceus" if task_name.startswith("ceus_") else "bmode"
    modality_prompt = torch.tensor(modality_prompt_one_hot_dict[modality_name], dtype=torch.float32)
    mode_prompt = torch.tensor(mode_prompt_one_hot_dict["whole"], dtype=torch.float32)
    type_prompt = torch.tensor(
        type_prompt_one_hot_dict[_safe_prompt_lookup(type_prompt_dict, dataset_name, "tumor")],
        dtype=torch.float32,
    )
    position_prompt = torch.tensor(
        position_prompt_one_hot_dict[_safe_prompt_lookup(position_prompt_dict, dataset_name, "indis")],
        dtype=torch.float32,
    )
    task_prompt = torch.tensor(task_prompt_one_hot_dict[prompt_name], dtype=torch.float32)
    return {
        "modality_prompt": modality_prompt,
        "mode_prompt": mode_prompt,
        "type_prompt": type_prompt,
        "position_prompt": position_prompt,
        "task_prompt": task_prompt,
    }


def _ensure_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if ext:
        return path
    return base + suffix


def _resolve_ceus_video_path(root: str, rel: str) -> str:
    rel = rel.replace("\\", "/").strip()
    candidates = []
    for prefix in ("", "ceus", "videos"):
        if prefix:
            candidates.append(os.path.join(root, prefix, rel).replace("\\", "/"))
        else:
            candidates.append(os.path.join(root, rel).replace("\\", "/"))
    for candidate in candidates:
        candidate = _ensure_suffix(candidate, ".npy")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot resolve CEUS video path '{rel}' under '{root}'")


def _parse_label_token(token: str) -> Optional[int]:
    token = str(token).strip().lower()
    if token in CLASS_NAME_TO_LABEL:
        return CLASS_NAME_TO_LABEL[token]
    try:
        return int(token)
    except Exception:
        return None


def _parse_ceus_classification_record(record: str) -> Tuple[str, Optional[int]]:
    raw = record.strip().replace("\\", "/")
    if not raw:
        raise ValueError("Empty CEUS classification record")

    for sep in ("\t", ","):
        parts = [part.strip() for part in raw.split(sep) if part.strip()]
        if len(parts) >= 2:
            label = _parse_label_token(parts[-1])
            if label is not None:
                return sep.join(parts[:-1]), label

    whitespace_parts = raw.split()
    if len(whitespace_parts) >= 2:
        label = _parse_label_token(whitespace_parts[-1])
        if label is not None:
            return " ".join(whitespace_parts[:-1]), label

    parent_token = raw.split("/")[-2] if "/" in raw else ""
    label = _parse_label_token(parent_token)
    return raw, label


# ---------------------- 单任务数据集定义 ---------------------- #

class SegmentationDataset2D(Dataset):
    """
    2D segmentation:
        base_dir/
          segmentation/
            DATASET_NAME/
              train.txt / val.txt / test.txt
              imgs/  
              masks/
              config.yaml  (label 映射)

    txt : 
        xxx.png
        imgs/xxx.png
        xxx             

    __getitem__ :
        {
          "task": "seg",
          "dataset": dataset_name,
          "image": Tensor(C,H,W) float32, [0,1]
          "mask":  Tensor(H,W)   long
          "case_name": 
        }
    """

    def __init__(
            self,
            base_dir: str,
            dataset_name: str,
            split: str = "train",
            transform=None,
            binary: bool = True,
            foreground_label: int = 1,
            joint_transform: Callable = None,
            img_size=256,
    ):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.binary = binary
        self.foreground_label = foreground_label

        self.root = os.path.join(base_dir, "segmentation", dataset_name)
        self.root = self.root.replace("\\", "/")
        txt_path = os.path.join(self.root, f"{split}.txt")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"segmentation txt not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            self.ids = [ln.strip() for ln in f.readlines() if ln.strip()]

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self.ids[idx]
        img_path = _auto_complete_path(self.root, rel, default_subdir="imgs")
        mask_path = _auto_complete_path(self.root, rel, default_subdir="masks")

        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        mask = _parse_seg_config_and_map_mask(self.base_dir, self.dataset_name, mask)

        if self.binary:
            mask = np.where(mask == self.foreground_label, 1, 0).astype(np.uint8)

        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)

        if 'train' in self.split:
            pt, point_label = random_click(np.array(mask), 1)
            bbox = random_bbox(np.array(mask), 1, self.img_size)
        else:
            pt, point_label = fixed_click(np.array(mask), 1)
            bbox = fixed_bbox(np.array(mask), 1, self.img_size)
        point_labels = np.array(point_label)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        prompt_dict = _build_task_prompts(self.dataset_name, "bmode_img_seg")

        sample = {
            "task": "bmode_img_seg",
            "dataset": self.dataset_name,
            "image": image,
            "mask": mask,
            "low_mask": low_mask,
            "label": int(0),
            "case_name": rel,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            **prompt_dict,
        }

        return sample


class ClassificationDataset2D(Dataset):
    """
    2D classification:
        base_dir/
          classification/
            DATASET_NAME/
              train.txt / val.txt / test.txt
              <class_id>/<img_name>.png 

    txt: DATASET_NAME:
        0/img0001.png
        1/img0002.png

    __getitem__ :
        {
          "task": "cls",
          "dataset": dataset_name,
          "image": Tensor(C,H,W) float32, [0,1]
          "label": int,
          "case_name": 相对路径
        }
    """

    def __init__(
            self,
            base_dir: str,
            dataset_name: str,
            split: str = "train",
            transform=None,
            joint_transform: Callable = None,
            img_size=256,
    ):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.root = os.path.join(base_dir, "classification", dataset_name).replace("\\", "/")
        txt_path = os.path.join(self.root, f"{split}.txt").replace("\\", "/")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"classification txt not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            self.ids = [ln.strip() for ln in f.readlines() if ln.strip()]

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self.ids[idx]
        rel = rel.replace("\\", "/")

        img_path = os.path.join(self.root, rel).replace("\\", "/")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path does not exist: {img_path}")

        image = cv2.imread(img_path, 0)
        # label = 父文件夹名字（假设是整数）
        try:
            label = int(rel.split("/")[-2])
        except Exception:
            # 如果不是 int，你可以在这里改成别的规则
            raise ValueError(f"Cannot parse class label from path: {rel}")
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)

        pt, point_label = fixed_click(np.array(mask), 1)
        bbox = fixed_bbox(np.array(mask), 1, self.img_size)
        point_labels = np.array(point_label)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        prompt_dict = _build_task_prompts(self.dataset_name, "bmode_img_cls")
        sample = {
            "task": "bmode_img_cls",
            "dataset": self.dataset_name,
            "image": image,
            "mask": mask,
            "low_mask": low_mask,
            "label": int(label),
            "case_name": rel,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            **prompt_dict,
        }
        return sample


class VideoSegmentationDataset(Dataset):
    """
    Video segmentation, 目录结构约定:
        base_dir/
          videos segmentation/
            DATASET_NAME/
              videos/
                train/xxx.npy
                val/xxx.npy
              annotations/
                train/xxx.npz
                val/xxx.npz

    xxx.npy: 形状可以是
             (T,H,W)      灰度
             (T,H,W,C)    彩色
    xxx.npz: 建议至少有一个 key 包含 mask:
             "masks" 或 "labels" 或 "label"

    __getitem__ 返回:
        {
          "task": "vid",
          "dataset": dataset_name,
          "video": Tensor(T,C,H,W) float32, [0,1]
          "mask":  Tensor(T,H,W)   long    (如果找到的话，否则全 0)
          "case_name": 文件名
        }

    真正数据格式你后续可以自己改，只要把这里的解析逻辑改掉即可。
    """

    def __init__(
            self,
            base_dir: str,
            dataset_name: str,
            split: str = "train",
            transform=None,
            frame_length: Optional[int] = None,
            ignore_index: int = 255,
    ):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.frame_length = frame_length
        self.ignore_index = ignore_index
        self.point_numbers = 1
        self.root = str(
            find_dataset_root(
                base_dir,
                dataset_name,
                task_directories=("video_seg", "videos segmentation"),
            )
        )
        split_entries = assert_disjoint_splits(self.root)
        self.video_paths = []
        self.annotation_paths = []
        for entry in split_entries[split]:
            self.video_paths.append(
                str(resolve_partitioned_file(self.root, "videos", split, entry, suffix=".npy"))
            )
            self.annotation_paths.append(
                str(resolve_partitioned_file(self.root, "annotations", split, entry, suffix=".npz"))
            )
        if transform:
            self.transform = transform
        else:
            # to_tensor = T.ToTensor()
            to_tensor = torch.tensor
            self.transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        ann_path = self.annotation_paths[idx]

        # ---- load video -> (T,H,W,3) ----
        image, mask, ef, edv, esv, spacing = load_video_and_mask_file(
            img_path=video_path,
            anno_path=ann_path,
            frame_length=self.frame_length)
        classes = 2
        if classes == 2:
            mask[mask > 1] = 0

        if self.transform:
            image, mask = self.transform(image, mask)
        # print("==== MASK DEBUG (NUMPY) ====")
        # print("mask type:", type(mask))
        # print("mask dtype:", mask.dtype)
        # print("mask shape:", mask.shape)
        # print("mask unique (first 10):", np.unique(mask)[:10])
        # print("mask min / max:", mask.min(), mask.max())
        # print("============================")
        pts, point_labels = [], []
        pt = []
        class_id = 1
        if 'train' in self.split:
            bbox = random_bbox(np.array(mask), class_id, 256)
            for mask_ in mask:
                pt, point_label = random_click(np.array(mask_), class_id)
                if self.point_numbers > 1:
                    for i in range(1, self.point_numbers):
                        _pt, _point_label = random_click(np.array(mask_), class_id)
                        pt = np.concatenate([pt, _pt], axis=0)
                        point_label = np.concatenate([point_label, _point_label], axis=0)
                pts.append(pt)
                point_labels.append(point_label)
        else:
            bbox = fixed_bbox(np.array(mask), class_id, 256)
            for mask_ in mask:
                pt, point_label = fixed_click(np.array(mask_), class_id)
                if self.point_numbers > 1:
                    for i in range(1, self.point_numbers):
                        _pt, _point_label = fixed_click(np.array(mask_), class_id)
                        pt = np.concatenate([pt, _pt], axis=0)
                        point_label = np.concatenate([point_label, _point_label], axis=0)
                pts.append(pt)
                point_labels.append(point_label)
        pt = np.stack(pts)
        point_label = np.stack(point_labels)

        prompt_dict = _build_task_prompts(self.dataset_name, "bmode_vid_seg")

        sample = {
            "task": "bmode_vid_seg",
            "dataset": self.dataset_name,
            "image": image,
            "mask": mask,
            "low_mask": mask,
            "label": int(0),
            "case_name": os.path.basename(video_path),
            'pt': pt,
            'bbox': bbox,
            'p_label': point_labels,
            'ef': ef,
            'edv': edv,
            'esv': esv,
            'spacing': spacing,
            **prompt_dict,
        }

        return sample


class CEUSVideoSegmentationDataset(Dataset):
    """CEUS video segmentation dataset.

    User-specified format (dataset_name = A):
        <root>/<A>/
            ceus/ *.npy        (video)
            mask/ *.npz        (10 frames mask)
            train.txt / val.txt / test.txt

    Processing (as requested):
        - Cut left/right half (along width).
        - Stack the two halves into channels: [left, right, 0].
        - GT mask: pick the half that has labels, discard the all-black half.

    Output keys follow existing video dataset to plug into current training.
    """

    def __init__(
            self,
            base_dir: str,
            dataset_name: str,
            split: str = "train",
            transform=None,
            frame_length: int = 10,
            prefer_right_if_empty: bool = True,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.frame_length = frame_length
        self.prefer_right_if_empty = prefer_right_if_empty
        self.point_numbers = 1

        self.root = str(
            find_dataset_root(
                base_dir,
                dataset_name,
                task_directories=("ceus_seg", "ceus segmentation"),
            )
        )
        split_entries = assert_disjoint_splits(self.root)
        self.ids = split_entries[split]
        self.video_collection = "videos" if os.path.isdir(os.path.join(self.root, "videos")) else "ceus"
        self.annotation_collection = "annotations" if os.path.isdir(os.path.join(self.root, "annotations")) else "mask"

    @staticmethod
    def _load_video_npy(path: str) -> np.ndarray:
        vid = np.load(path, allow_pickle=True)
        if vid.ndim == 3:
            # (T,H,W) or (H,W,T)
            if vid.shape[0] > 64 and vid.shape[-1] <= 64:
                vid = np.transpose(vid, (2, 0, 1))
            vid = vid[..., None]
        elif vid.ndim == 4:
            # (T,H,W,C) or (H,W,C,T)
            if vid.shape[-1] <= 64 and vid.shape[-1] < vid.shape[0]:
                vid = np.transpose(vid, (3, 0, 1, 2))
        else:
            raise ValueError(f"Unsupported video shape {vid.shape} for {path}")

        # vid = vid.astype(np.float32)
        # vid = (vid - vid.mean()) / (vid.std() + 1e-6)

        return vid  # (T,H,W,C)

    @staticmethod
    def _load_mask_npz(path: str) -> np.ndarray:
        z = np.load(path, allow_pickle=True)
        for k in ["mask", "masks", "label", "labels", "arr_0"]:
            if k in z.files:
                m = z[k]
                break
        else:
            if len(z.files) == 0:
                raise ValueError(f"Empty npz: {path}")
            m = z[z.files[0]]
        m = np.asarray(m)
        if m.ndim == 2:
            m = m[None, ...]
        if m.ndim != 3:
            raise ValueError(f"Unsupported mask shape {m.shape} for {path}")
        return (m > 0).astype(np.uint8)

    @staticmethod
    def _cut_stack_and_pick_gt(video: np.ndarray, mask: np.ndarray, prefer_right_if_empty: bool = True,
                               baseline_frames: int = 5, ):
        C, T, H, W = video.shape
        mid = W // 2

        if C == 1:
            gray = video[0]
        else:
            gray = 0.2989 * video[0] + 0.5870 * video[1] + 0.1140 * video[2]

        left = gray[:, :, :mid]
        right = gray[:, :, mid:mid + mid]

        mT, mH, mW = mask.shape
        if (mH != H) or (mW != W):
            resized = [cv2.resize(mask[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) for i in range(mT)]
            mask = np.stack(resized, axis=0)
        mask = mask[:T]

        m_left = mask[:, :, :mid]
        m_right = mask[:, :, mid:mid + mid]

        sum_left = int(m_left.sum())
        sum_right = int(m_right.sum())
        if sum_left == 0 and sum_right == 0:
            use_right = bool(prefer_right_if_empty)
        else:
            use_right = sum_right >= sum_left

        gt = m_right if use_right else m_left
        bmode = right if use_right else left
        ceus = left if use_right else right
        bf = max(1, min(int(baseline_frames), T))
        baseline = ceus[:bf].mean(axis=0, keepdims=True)  # (1,H,mid)
        ceus_minus_base = ceus - baseline  # (T,H,mid)
        img = np.stack([bmode, ceus, ceus_minus_base], axis=-1)
        img = img.transpose(0, 3, 1, 2)
        return img.astype(np.float32), gt.astype(np.uint8)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self.ids[idx].replace("\\", "/")
        base = os.path.splitext(os.path.basename(rel))[0]

        video_path = str(resolve_partitioned_file(self.root, self.video_collection, self.split, rel, suffix=".npy"))
        mask_path = str(resolve_partitioned_file(self.root, self.annotation_collection, self.split, rel, suffix=".npz"))

        video = self._load_video_npy(video_path)
        mask = self._load_mask_npz(mask_path)

        if self.frame_length is not None and self.frame_length > 0:
            # _load_video_npy canonicalizes supported layouts to (C,T,H,W).
            T0 = video.shape[1]
            if T0 >= self.frame_length:
                inds = np.linspace(0, T0 - 1, self.frame_length).round().astype(int)
                video = video[:, inds, ...]
            else:
                pad = np.repeat(video[:, -1:, ...], self.frame_length - T0, axis=1)
                video = np.concatenate([video, pad], axis=1)

            M0 = mask.shape[0]
            if M0 >= self.frame_length:
                inds = np.linspace(0, M0 - 1, self.frame_length).round().astype(int)
                mask = mask[inds]
            else:
                mask = np.concatenate([mask, np.repeat(mask[-1:], self.frame_length - M0, axis=0)], axis=0)

        image, mask = self._cut_stack_and_pick_gt(video, mask, prefer_right_if_empty=self.prefer_right_if_empty)

        if self.transform:
            image, mask = self.transform(image, mask)

        pts, point_labels = [], []
        class_id = 1
        if 'train' in self.split:
            bbox = random_bbox(np.array(mask), class_id, 256)
            for mask_ in mask:
                pt, point_label = random_click(np.array(mask_), class_id)
                pts.append(pt)
                point_labels.append(point_label)
        else:
            bbox = fixed_bbox(np.array(mask), class_id, 256)
            for mask_ in mask:
                pt, point_label = fixed_click(np.array(mask_), class_id)
                pts.append(pt)
                point_labels.append(point_label)
        pt = np.stack(pts)
        point_label = np.stack(point_labels)

        prompt_dict = _build_task_prompts(self.dataset_name, "ceus_vid_seg")

        sample = {
            "task": "ceus_vid_seg",
            "dataset": self.dataset_name,
            "image": image,
            "mask": mask,
            "low_mask": mask,
            "label": int(0),
            "case_name": base,
            'pt': pt,
            'bbox': bbox,
            'p_label': point_label,
            **prompt_dict,
        }
        return sample


class CEUSVideoClassificationDataset(Dataset):
    def __init__(
            self,
            base_dir: str,
            dataset_name: str,
            split: str = "train",
            transform=None,
            frame_length: int = 10,
            prefer_right_if_empty: bool = True,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.frame_length = frame_length
        self.prefer_right_if_empty = prefer_right_if_empty

        self.root = str(
            find_dataset_root(
                base_dir,
                dataset_name,
                task_directories=("ceus_cls", "ceus classification"),
            )
        )
        split_entries = assert_disjoint_splits(self.root)
        self.ids = split_entries[split]

        self.label_lookup = self._load_label_lookup()

    def _load_label_lookup(self) -> Dict[str, int]:
        candidates = [
            os.path.join(self.root, "labels.csv"),
            os.path.join(self.root, "label.csv"),
            os.path.join(self.root, "metadata.csv"),
            os.path.join(self.root, "labels.xlsx"),
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if path.endswith(".xlsx"):
                    table = pd.read_excel(path)
                else:
                    table = pd.read_csv(path)
            except Exception:
                continue

            lowered = {str(column).strip().lower(): column for column in table.columns}
            id_col = None
            label_col = None
            for candidate in ("case_name", "case", "id", "video", "filename", "name"):
                if candidate in lowered:
                    id_col = lowered[candidate]
                    break
            for candidate in ("label", "class", "target", "malignancy", "y"):
                if candidate in lowered:
                    label_col = lowered[candidate]
                    break
            if id_col is None or label_col is None:
                continue

            mapping: Dict[str, int] = {}
            for _, row in table.iterrows():
                label = _parse_label_token(row[label_col])
                if label is None:
                    continue
                key = os.path.splitext(os.path.basename(str(row[id_col])))[0]
                mapping[key] = label
            if mapping:
                return mapping
        return {}

    @staticmethod
    def _cut_stack_without_mask(video: np.ndarray, prefer_right_if_empty: bool = True, baseline_frames: int = 5):
        C, T, H, W = video.shape
        mid = W // 2
        if C == 1:
            gray = video[0]
        else:
            gray = 0.2989 * video[0] + 0.5870 * video[1] + 0.1140 * video[2]

        left = gray[:, :, :mid]
        right = gray[:, :, mid:mid + mid]
        bmode = right if prefer_right_if_empty else left
        ceus = left if prefer_right_if_empty else right
        bf = max(1, min(int(baseline_frames), T))
        baseline = ceus[:bf].mean(axis=0, keepdims=True)
        ceus_minus_base = ceus - baseline
        img = np.stack([bmode, ceus, ceus_minus_base], axis=-1)
        return img.transpose(0, 3, 1, 2).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_record = self.ids[idx]
        rel, parsed_label = _parse_ceus_classification_record(raw_record)
        video_path = _resolve_ceus_video_path(self.root, rel)
        base = os.path.splitext(os.path.basename(video_path))[0]

        label = parsed_label
        if label is None:
            label = self.label_lookup.get(base, None)
        if label is None:
            raise ValueError(f"Cannot resolve CEUS classification label for record '{raw_record}'")

        video = CEUSVideoSegmentationDataset._load_video_npy(video_path)
        if self.frame_length is not None and self.frame_length > 0:
            total_frames = video.shape[1]
            if total_frames >= self.frame_length:
                inds = np.linspace(0, total_frames - 1, self.frame_length).round().astype(int)
                video = video[:, inds, ...]
            else:
                pad = np.repeat(video[:, -1:, ...], self.frame_length - total_frames, axis=1)
                video = np.concatenate([video, pad], axis=1)

        mask_path = os.path.join(self.root, "mask", base + ".npz").replace("\\", "/")
        if os.path.isfile(mask_path):
            mask = CEUSVideoSegmentationDataset._load_mask_npz(mask_path)
            if mask.shape[0] >= self.frame_length:
                inds = np.linspace(0, mask.shape[0] - 1, self.frame_length).round().astype(int)
                mask = mask[inds]
            else:
                pad = np.repeat(mask[-1:], self.frame_length - mask.shape[0], axis=0)
                mask = np.concatenate([mask, pad], axis=0)
            image, mask = CEUSVideoSegmentationDataset._cut_stack_and_pick_gt(
                video,
                mask,
                prefer_right_if_empty=self.prefer_right_if_empty,
            )
        else:
            image = self._cut_stack_without_mask(video, prefer_right_if_empty=self.prefer_right_if_empty)
            mask = np.zeros((image.shape[0], image.shape[-2], image.shape[-1]), dtype=np.uint8)

        if self.transform:
            image, mask = self.transform(image, mask)

        pts = []
        point_labels = []
        bbox = fixed_bbox(np.array(mask), 1, 256)
        for frame_mask in mask:
            point, point_label = fixed_click(np.array(frame_mask), 1)
            pts.append(point)
            point_labels.append(point_label)
        pt = np.stack(pts)
        point_label = np.stack(point_labels)
        prompt_dict = _build_task_prompts(self.dataset_name, "ceus_vid_cls")

        sample = {
            "task": "ceus_vid_cls",
            "dataset": self.dataset_name,
            "image": image,
            "mask": mask,
            "low_mask": mask,
            "label": int(label),
            "case_name": base,
            "pt": pt,
            "bbox": bbox,
            "p_label": point_label,
            **prompt_dict,
        }
        return sample


# ---------------------- 总的统一数据集：dataset_new ---------------------- #

class dataset_ce(Dataset):
    """
    统一数据集：
    - 支持 2D segmentation / 2D classification / video segmentation
    - 自动扫描有哪些子数据集（也可以手动指定 list）

    __getitem__ 返回的 sample 一定包含 key: "task"
        - "seg": 有 "image"(C,H,W), "mask"(H,W)
        - "cls": 有 "image"(C,H,W), "label"
        - "vid": 有 "video"(T,C,H,W), "mask"(T,H,W)
    以及:
        - "dataset": 数据集名字
        - "case_name": 数据内部 ID
    """

    def __init__(
            self,
            base_dir: str,
            split: str = "train",
            img_size=256,
            include_seg: bool = True,
            include_cls: bool = True,
            include_vid: bool = True,
            include_ceus_vid: bool = False,
            include_ceus_cls: bool = False,
            seg_datasets: Optional[List[str]] = None,
            cls_datasets: Optional[List[str]] = None,
            vid_datasets: Optional[List[str]] = None,
            ceus_vid_datasets: Optional[List[str]] = None,
            ceus_cls_datasets: Optional[List[str]] = None,
            seg_transform: Callable = None,
            cls_transform: Callable = None,
            vid_transform: Callable = None,
            ceus_vid_transform: Callable = None,
            ceus_cls_transform: Callable = None,
            frame_length: Optional[int] = None,
            ceus_frame_length: int = 10,
            binary_seg: bool = True,
            foreground_label: int = 1,
            joint_transform: Callable = None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.seg_transform = seg_transform
        self.cls_transform = cls_transform
        self.vid_transform = vid_transform
        self.ceus_vid_transform = ceus_vid_transform if ceus_vid_transform is not None else vid_transform
        self.ceus_cls_transform = ceus_cls_transform if ceus_cls_transform is not None else self.ceus_vid_transform
        self.frame_length = frame_length
        self.ceus_frame_length = ceus_frame_length
        self.binary_seg = binary_seg
        self.foreground_label = foreground_label

        self.seg_datasets = seg_datasets
        self.cls_datasets = cls_datasets
        self.vid_datasets = vid_datasets
        self.ceus_vid_datasets = ceus_vid_datasets
        self.ceus_cls_datasets = ceus_cls_datasets

        self.seg_subsets: List[SegmentationDataset2D] = []
        self.cls_subsets: List[ClassificationDataset2D] = []
        self.vid_subsets: List[VideoSegmentationDataset] = []
        self.ceus_vid_subsets: List[CEUSVideoSegmentationDataset] = []
        self.ceus_cls_subsets: List[CEUSVideoClassificationDataset] = []
        self.index_map: List[Tuple[str, int, int]] = []
        # index_map 的每个元素是 (task_type, subset_idx, local_idx)

        # ---------- 1. segmentation ---------- #
        if include_seg:
            if self.seg_datasets is None:
                self.seg_datasets = []
            for ds_name in self.seg_datasets:
                subset = SegmentationDataset2D(
                    base_dir=base_dir,
                    dataset_name=ds_name,
                    split=split,
                    transform=self.seg_transform,
                    binary=binary_seg,
                    foreground_label=foreground_label,
                    joint_transform=joint_transform,
                    img_size=self.img_size,
                )
                subset_idx = len(self.seg_subsets)
                self.seg_subsets.append(subset)
                for local_idx in range(len(subset)):
                    self.index_map.append(("seg", subset_idx, local_idx))

        # ---------- 2. classification ---------- #
        if include_cls:
            if self.cls_datasets is None:
                self.cls_datasets = []
            for ds_name in self.cls_datasets:
                subset = ClassificationDataset2D(
                    base_dir=base_dir,
                    dataset_name=ds_name,
                    split=split,
                    transform=self.cls_transform,
                    joint_transform=joint_transform,
                    img_size=self.img_size,
                )
                subset_idx = len(self.cls_subsets)
                self.cls_subsets.append(subset)
                for local_idx in range(len(subset)):
                    self.index_map.append(("cls", subset_idx, local_idx))

        # ---------- 3. video segmentation ---------- #
        if include_vid:
            if self.vid_datasets is None:
                self.vid_datasets = []
            for ds_name in self.vid_datasets:
                subset = VideoSegmentationDataset(
                    base_dir=base_dir,
                    dataset_name=ds_name,
                    split=split,
                    transform=self.vid_transform,
                    frame_length=self.frame_length,
                )
                subset_idx = len(self.vid_subsets)
                self.vid_subsets.append(subset)
                for local_idx in range(len(subset)):
                    self.index_map.append(("bmode_vid_seg", subset_idx, local_idx))

        # ---------- 4. CEUS video segmentation ---------- #
        if include_ceus_vid:
            if self.ceus_vid_datasets is None:
                self.ceus_vid_datasets = []
            for ds_name in self.ceus_vid_datasets:
                subset = CEUSVideoSegmentationDataset(
                    base_dir=base_dir,
                    dataset_name=ds_name,
                    split=split,
                    transform=self.ceus_vid_transform,
                    frame_length=self.ceus_frame_length,
                )
                subset_idx = len(self.ceus_vid_subsets)
                self.ceus_vid_subsets.append(subset)
                for local_idx in range(len(subset)):
                    self.index_map.append(("ceus_vid_seg", subset_idx, local_idx))

        # ---------- 5. CEUS video classification ---------- #
        if include_ceus_cls:
            if self.ceus_cls_datasets is None:
                self.ceus_cls_datasets = []
            for ds_name in self.ceus_cls_datasets:
                subset = CEUSVideoClassificationDataset(
                    base_dir=base_dir,
                    dataset_name=ds_name,
                    split=split,
                    transform=self.ceus_cls_transform,
                    frame_length=self.ceus_frame_length,
                )
                subset_idx = len(self.ceus_cls_subsets)
                self.ceus_cls_subsets.append(subset)
                for local_idx in range(len(subset)):
                    self.index_map.append(("ceus_vid_cls", subset_idx, local_idx))

        if len(self.index_map) == 0:
            raise RuntimeError(
                f"No data found. base_dir={base_dir}, split={split}, "
                f"include_seg={include_seg}, include_cls={include_cls}, include_vid={include_vid}, "
                f"include_ceus_vid={include_ceus_vid}, include_ceus_cls={include_ceus_cls}"
            )

    # ---------- 自动扫描子数据集 ---------- #

    def _auto_find_seg_datasets(self, split: str) -> List[str]:
        root = os.path.join(self.base_dir, "segmentation")
        if not os.path.isdir(root):
            return []
        ds_list = []
        for d in sorted(os.listdir(root)):
            sub_dir = os.path.join(root, d)
            if not os.path.isdir(sub_dir):
                continue
            txt_path = os.path.join(sub_dir, f"{split}.txt")
            if os.path.isfile(txt_path):
                ds_list.append(d)
        return ds_list

    def _auto_find_cls_datasets(self, split: str) -> List[str]:
        root = os.path.join(self.base_dir, "classification")
        if not os.path.isdir(root):
            return []
        ds_list = []
        for d in sorted(os.listdir(root)):
            sub_dir = os.path.join(root, d)
            if not os.path.isdir(sub_dir):
                continue
            txt_path = os.path.join(sub_dir, f"{split}.txt")
            if os.path.isfile(txt_path):
                ds_list.append(d)
        return ds_list

    def _auto_find_vid_datasets(self, split: str) -> List[str]:
        ds_list = []
        for task_dir in ("video_seg", "videos segmentation"):
            root = os.path.join(self.base_dir, task_dir)
            if not os.path.isdir(root):
                continue
            for d in sorted(os.listdir(root)):
                dataset_root = os.path.join(root, d)
                if os.path.isdir(dataset_root) and split_is_available(dataset_root, split):
                    ds_list.append(d)
        return list(dict.fromkeys(ds_list))

    # ---------- Dataset 接口 ---------- #

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_type, subset_idx, local_idx = self.index_map[idx]
        if task_type == "seg":
            return self.seg_subsets[subset_idx][local_idx]
        elif task_type == "cls":
            return self.cls_subsets[subset_idx][local_idx]
        elif task_type == "b-mode video se" or task_type == "bmode_vid_seg":
            return self.vid_subsets[subset_idx][local_idx]
        elif task_type == "ceus video se" or task_type == "ceus_vid_seg":
            return self.ceus_vid_subsets[subset_idx][local_idx]
        elif task_type == "ceus_vid_cls":
            return self.ceus_cls_subsets[subset_idx][local_idx]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
