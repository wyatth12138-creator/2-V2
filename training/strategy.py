ENCODER_ADAPTER_KEYS = (
    "cnn_embed",
    "post_pos_embed",
    "Adapter",
    "rel_pos",
    "upneck",
    "enc_modality_film",
    "enc_mode_film",
    "enc_pos_bias",
    "enc_type_film",
    "enc_task_film",
)

SEGMENTATION_MODULES = (
    "mask_decoder",
    "prototype_extractor",
    "dynamic_prompt_generator",
    "task_interaction",
    "memory_attention",
    "memory_encoder",
    "prompt_bank",
)

CLASSIFICATION_MODULES = (
    "cls_head",
    "fla_score",
    "temporal_observer",
)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = flag


def _enable_module(model, module_name: str) -> None:
    module = getattr(model, module_name, None)
    if module is None:
        return
    set_requires_grad(module, True)
    module.train()


def _enable_encoder_adapters(model) -> None:
    if not hasattr(model, "image_encoder"):
        return
    encoder = model.image_encoder
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = any(key in name for key in ENCODER_ADAPTER_KEYS)
    encoder.train()


def set_stage(model, stage: str):
    """Apply strict segmentation-then-classification freezing."""

    model = unwrap_model(model)
    stage = stage.lower()
    set_requires_grad(model, False)
    model.eval()

    if stage in ("segmentation", "stage_1", "stage1"):
        _enable_encoder_adapters(model)
        for module_name in SEGMENTATION_MODULES:
            _enable_module(model, module_name)
        if hasattr(model, "prompt_encoder"):
            model.prompt_encoder.eval()
        return

    if stage in ("classification", "stage_2", "stage2"):
        for module_name in CLASSIFICATION_MODULES:
            _enable_module(model, module_name)
        # Only task-specific prompt rows need to condition the frozen encoder
        # for the two classification task types. Other SSCP banks stay frozen.
        if hasattr(model, "prompt_bank") and hasattr(model.prompt_bank, "W_task"):
            model.prompt_bank.W_task.requires_grad = True
        return

    raise ValueError(f"Unknown two-stage training phase: {stage}")


def build_optimizer(model, lr, stage: str, weight_decay=0.0):
    import torch

    model = unwrap_model(model)
    stage = stage.lower()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError(f"No trainable parameters for stage {stage!r}")

    if stage in ("classification", "stage_2", "stage2"):
        task_prompt = [parameter for name, parameter in named_parameters if name == "prompt_bank.W_task"]
        classifier = [parameter for name, parameter in named_parameters if name != "prompt_bank.W_task"]
        groups = []
        if classifier:
            groups.append({"params": classifier, "lr": lr, "lr_scale": 1.0, "weight_decay": weight_decay})
        if task_prompt:
            # Zero weight decay prevents unused segmentation prompt rows from
            # changing when only classification rows receive gradients.
            groups.append({"params": task_prompt, "lr": lr, "lr_scale": 1.0, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    backbone_params = []
    head_params = []
    for name, parameter in named_parameters:
        if name.startswith("image_encoder."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * 0.5, "lr_scale": 0.5})
    if head_params:
        groups.append({"params": head_params, "lr": lr, "lr_scale": 1.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)


def step_lr(args, optimizer, iter_num, max_iterations):
    if args.warmup and iter_num < args.warmup_period:
        lr = args.base_lr * ((iter_num + 1) / args.warmup_period)
    else:
        shift_iter = max(iter_num - args.warmup_period, 0)
        lr = args.base_lr * (1.0 - shift_iter / max(max_iterations, 1)) ** 0.9
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))
    return iter_num + 1
