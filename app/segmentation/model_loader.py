import torch
from app.segmentation.unet import UNet
from app.segmentation.deeplabv3 import load_deeplabv3


def load_model(model_type: str, model_path: str | None = None):

    if model_type.lower() == "unet":
        model = UNet()

        if model_path is None:
            raise ValueError("MODEL_PATH required for UNet")

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    elif model_type.lower() == "deeplabv3":
        # torchvision pretrained
        model = load_deeplabv3()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()
    return model
