import os

# This file is used to configure the training parameters for each task

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = os.environ.get("UIV_US_IMAGE_ROOT", ".")
    data_roots = {
        "bmode_img_seg": data_path,
        "bmode_img_cls": data_path,
        "bmode_vid_seg": os.environ.get(
            "UIV_US_VIDEO_ROOT",
            ".",
        ),
        "ceus_vid_seg": os.environ.get(
            "UIV_US_CEUS_ROOT",
            ".",
        ),
        "ceus_vid_cls": os.environ.get(
            "UIV_US_CEUS_ROOT",
            ".",
        ),
    }
    data_subpath = ""
    save_path = os.environ.get("UIV_US_OUTPUT_ROOT", "./self_checkpoints/UIV-US/")
    result_path = "./result/UIV-US/CCAUtsne1/"
    tensorboard_path = "./tensorboard/UIV-US/"
    load_path = save_path + ""
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 100                 # number of total epochs to run (default: 400)
    batch_size = 16               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "val"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 10               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "UIV-US"


# ==================================================================================================
def get_config(task="US30K"):
    if task == "CE":
        return Config_TN3K()
    else:
        raise ValueError(f"Unsupported task configuration: {task}")

