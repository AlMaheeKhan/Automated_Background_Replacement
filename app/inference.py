import cv2
import torch
import numpy as np
from app.config import IMAGE_SIZE
from app.utils.mask_utils import apply_mask

def run_inference(model, image_path, background_path=None):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, IMAGE_SIZE)

    tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(tensor)

    mask = output.squeeze().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    return apply_mask(image_resized, mask, background_path)
