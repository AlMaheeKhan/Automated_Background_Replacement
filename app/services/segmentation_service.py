import cv2
import torch
import numpy as np

from app.config import IMAGE_SIZE, MODEL_TYPE, MODEL_PATH
from app.segmentation.model_loader import load_model
from app.utils.mask_utils import apply_mask


class SegmentationService:
    def __init__(self):
        # Load model once during startup (CPU only)
        self.model = load_model(MODEL_TYPE, MODEL_PATH)
        self.model.eval()

    def process(self, image_bytes: bytes, background_bytes: bytes = None) -> bytes:
        """
        Main pipeline:
        1. Decode input image
        2. Preprocess
        3. Run segmentation model
        4. Extract mask (UNet or DeepLabV3)
        5. Apply background replacement
        6. Encode final PNG
        """

        # -------------------------------------------------
        # 1️⃣ Decode input image safely
        # -------------------------------------------------
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid input image. Could not decode.")

        image_resized = cv2.resize(image, IMAGE_SIZE)

        # -------------------------------------------------
        # 2️⃣ Preprocess
        # -------------------------------------------------
        tensor = (
            torch.from_numpy(image_resized)
            .permute(2, 0, 1)      # HWC -> CHW
            .unsqueeze(0)         # Add batch dimension
            .float() / 255.0
        )

        # -------------------------------------------------
        # 3️⃣ Inference
        # -------------------------------------------------
        with torch.no_grad():
            output = self.model(tensor)

        # torchvision DeepLab returns dict
        if isinstance(output, dict):
            output = output["out"]

        # -------------------------------------------------
        # 4️⃣ Extract Mask
        # -------------------------------------------------
        if MODEL_TYPE == "deeplabv3":
            # DeepLab output shape: [1, 21, H, W]
            output = output.squeeze(0)  # remove batch

            # Take class with highest probability
            mask = torch.argmax(output, dim=0).byte().cpu().numpy()

            # COCO class index for person = 15
            mask = (mask == 15).astype(np.uint8)

        else:
            # UNet binary segmentation
            mask = torch.sigmoid(output)
            mask = mask.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

        # -------------------------------------------------
        # 5️⃣ Decode background safely (optional)
        # -------------------------------------------------
        background = None

        if background_bytes is not None:
            bg_array = np.frombuffer(background_bytes, np.uint8)
            background = cv2.imdecode(bg_array, cv2.IMREAD_COLOR)

            if background is None:
                raise ValueError("Invalid background image.")

            background = cv2.resize(background, IMAGE_SIZE)

        # -------------------------------------------------
        # 6️⃣ Apply mask
        # -------------------------------------------------
        result = apply_mask(image_resized, mask, background)

        if result is None or result.size == 0:
            raise ValueError("Segmentation result is empty.")

        # Ensure correct dtype for OpenCV
        result = result.astype(np.uint8)

        # -------------------------------------------------
        # 7️⃣ Encode result
        # -------------------------------------------------
        success, encoded = cv2.imencode(".png", result)

        if not success:
            raise ValueError("Failed to encode result image.")

        return encoded.tobytes()
