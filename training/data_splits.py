from collections import OrderedDict, defaultdict
from copy import copy
import math

from torch.utils.data import DataLoader, Subset

from utils.data_us import JointTransform2D, JointTransform3D
from utils.dataset_ce_pat import dataset_ce


# Dataset groups define modality and storage; training stages are selected by task type.
DATA_SPLITS = {
    "bmode_train": {
        "bmode_img_seg": ["BUS-BRA", "TN3K", "MMOTU", "Fetal_HC", "kidneyUS_capsule", "EchoNet-Dynamic"],
        "bmode_img_cls": ["BUS-BRA", "TN3K", "Fatty-Liver", "Appendix"],
        "bmode_vid_seg": ["CAMUS"],
        "ceus_vid_seg": [],
        "ceus_vid_cls": [],
    },
    "bmode_val": {
        "bmode_img_seg": ["BUS-BRA", "TN3K", "MMOTU", "Fetal_HC", "kidneyUS_capsule", "EchoNet-Dynamic"],
        "bmode_img_cls": ["BUS-BRA", "TN3K", "Fatty-Liver", "Appendix"],
        "bmode_vid_seg": ["CAMUS"],
        "ceus_vid_seg": [],
        "ceus_vid_cls": [],
    },
    "ceus_train": {
        "bmode_img_seg": [],
        "bmode_img_cls": [],
        "bmode_vid_seg": [],
        "ceus_vid_seg": ["Breast", "Thyroid"],
        "ceus_vid_cls": ["Breast", "Liver", "Prostate", "Thyroid"],
    },
    "ceus_val": {
        "bmode_img_seg": [],
        "bmode_img_cls": [],
        "bmode_vid_seg": [],
        "ceus_vid_seg": ["Breast", "Thyroid"],
        "ceus_vid_cls": ["Breast", "Liver", "Prostate", "Thyroid"],
    },
}


TASK_ORDER = ["bmode_img_seg", "bmode_img_cls", "bmode_vid_seg", "bmode_vid_cls", "ceus_vid_seg", "ceus_vid_cls"]
DATA_SPLITS["bmode_train"]["bmode_vid_cls"] = []
DATA_SPLITS["bmode_val"]["bmode_vid_cls"] = []


def make_transforms(opt, args):
    tf_train_2d = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.0,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    tf_val_2d = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    tf_train_3d = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.0,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    tf_val_3d = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    return tf_train_2d, tf_val_2d, tf_train_3d, tf_val_3d


def dataset_name_for_index(dataset_obj, sample_index: int) -> str:
    task_type, subset_idx, _ = dataset_obj.index_map[sample_index]
    if task_type == "seg":
        return dataset_obj.seg_subsets[subset_idx].dataset_name
    if task_type == "cls":
        return dataset_obj.cls_subsets[subset_idx].dataset_name
    if task_type in ("bmode_vid_seg", "b-mode video se"):
        return dataset_obj.vid_subsets[subset_idx].dataset_name
    if task_type in ("ceus_vid_seg", "ceus video se"):
        return dataset_obj.ceus_vid_subsets[subset_idx].dataset_name
    if task_type == "ceus_vid_cls":
        return dataset_obj.ceus_cls_subsets[subset_idx].dataset_name
    raise ValueError(f"Unknown dataset index task_type: {task_type}")


def limit_dataset_for_fewshot(dataset_obj, fewshot_mode: str):
    if fewshot_mode == "all":
        return dataset_obj

    grouped_indices = defaultdict(list)
    if hasattr(dataset_obj, "datasets"):
        offset = 0
        for ds in dataset_obj.datasets:
            grouped_indices[ds.dataset_name].extend(range(offset, offset + len(ds)))
            offset += len(ds)
    else:
        for sample_index in range(len(dataset_obj)):
            grouped_indices[dataset_name_for_index(dataset_obj, sample_index)].append(sample_index)

    selected = []
    for _, indices in sorted(grouped_indices.items()):
        if fewshot_mode.endswith("shot"):
            limit = int(fewshot_mode.replace("shot", ""))
        elif fewshot_mode.endswith("pct"):
            ratio = float(fewshot_mode.replace("pct", "")) / 100.0
            limit = max(1, int(math.ceil(len(indices) * ratio)))
        else:
            raise ValueError(f"Unsupported fewshot mode: {fewshot_mode}")
        selected.extend(indices[: min(limit, len(indices))])
    return Subset(dataset_obj, selected)


def build_task_dataset(opt, task_name: str, datasets, split: str, tf_2d, tf_3d, args=None):
    if not datasets:
        return None
    data_roots = getattr(opt, "data_roots", {})
    if task_name == "bmode_vid_cls" or (
        task_name == "ceus_vid_cls" and getattr(args, "ordered_video_cls", False)
    ):
        from torch.utils.data import ConcatDataset
        from utils.ordered_video_cls import OrderedVideoClassificationDataset

        # JointTransform3D applies contrast independently per frame. That
        # would fabricate local intensity changes for the temporal observer.
        ordered_transform = copy(tf_3d)
        ordered_transform.p_contr = 0.0
        return ConcatDataset([
            OrderedVideoClassificationDataset(
                data_roots.get(task_name, opt.data_path), name, task_name, split, ordered_transform,
                frame_length=getattr(args, "video_frames", 10),
            ) for name in datasets
        ])
    kwargs = {
        "base_dir": data_roots.get(task_name, opt.data_path),
        "split": split,
        "img_size": tf_2d.img_size,
        "include_seg": False,
        "include_cls": False,
        "include_vid": False,
        "include_ceus_vid": False,
        "include_ceus_cls": False,
        "seg_datasets": [],
        "cls_datasets": [],
        "vid_datasets": [],
        "ceus_vid_datasets": [],
        "ceus_cls_datasets": [],
        "joint_transform": tf_2d,
        "vid_transform": tf_3d,
        "ceus_vid_transform": tf_3d,
        "ceus_cls_transform": tf_3d,
        "frame_length": 10,
        "ceus_frame_length": 10,
        "binary_seg": True,
        "foreground_label": 1,
    }
    if task_name == "bmode_img_seg":
        kwargs["include_seg"] = True
        kwargs["seg_datasets"] = datasets
    elif task_name == "bmode_img_cls":
        kwargs["include_cls"] = True
        kwargs["cls_datasets"] = datasets
    elif task_name == "bmode_vid_seg":
        kwargs["include_vid"] = True
        kwargs["vid_datasets"] = datasets
    elif task_name == "ceus_vid_seg":
        kwargs["include_ceus_vid"] = True
        kwargs["ceus_vid_datasets"] = datasets
    elif task_name == "ceus_vid_cls":
        kwargs["include_ceus_cls"] = True
        kwargs["ceus_cls_datasets"] = datasets
    else:
        raise ValueError(f"Unknown task for dataset creation: {task_name}")
    return dataset_ce(**kwargs)


def batch_size_for_task(args, task_name: str) -> int:
    if task_name in ("bmode_img_seg",):
        return args.bs_img
    if task_name in ("bmode_img_cls",):
        return args.batch_size
    return args.bs_vid


def build_loader_bundle(opt, task_to_datasets, split, args, tf_2d, tf_3d, fewshot_mode="all", shuffle=True):
    loaders = OrderedDict()
    for task_name in TASK_ORDER:
        datasets = task_to_datasets.get(task_name, [])
        dataset_obj = build_task_dataset(opt, task_name, datasets, split, tf_2d, tf_3d, args)
        if dataset_obj is None:
            continue
        dataset_obj = limit_dataset_for_fewshot(dataset_obj, fewshot_mode)
        if len(dataset_obj) == 0:
            continue
        loaders[task_name] = DataLoader(
            dataset_obj,
            batch_size=batch_size_for_task(args, task_name),
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return loaders


def build_domain_loader_bundle(opt, task_to_datasets, split, args, tf_2d, tf_3d, fewshot_mode="all", shuffle=True):
    """One homogeneous loader per (task,dataset), without changing legacy sampling.

    Keeping source datasets separate permits explicit within-task domain
    balance and same-state domain-gradient comparisons. A task identifier
    remains in each sample; the tuple key is source metadata, not a new task.
    Validation/test loaders should use shuffle=False and fewshot_mode='all'.
    """
    loaders = OrderedDict()
    for task_name in TASK_ORDER:
        for dataset_name in task_to_datasets.get(task_name, []):
            key = (task_name, dataset_name)
            if key in loaders:
                raise ValueError(f"Duplicate domain loader: {key}")
            dataset_obj = build_task_dataset(
                opt, task_name, [dataset_name], split, tf_2d, tf_3d, args
            )
            if dataset_obj is None:
                continue
            dataset_obj = limit_dataset_for_fewshot(dataset_obj, fewshot_mode)
            if len(dataset_obj) == 0:
                continue
            loaders[key] = DataLoader(
                dataset_obj,
                batch_size=batch_size_for_task(args, task_name),
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=True,
            )
    return loaders
