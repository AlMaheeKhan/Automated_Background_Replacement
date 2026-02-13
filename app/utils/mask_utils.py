import cv2
import numpy as np


def apply_mask(image, mask, background=None):
    """
    Applies segmentation mask and replaces background.
    image: original image (H, W, 3)
    mask: binary mask (H, W)
    background: background image (H, W, 3) or None
    """

    # Ensure mask is binary
    mask = mask.astype(np.uint8)

    # Resize background to match image
    if background is not None:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
    else:
        # Default background: white
        background = np.ones_like(image) * 255

    # Expand mask to 3 channels
    mask_3c = np.stack([mask]*3, axis=-1)

    # Foreground extraction
    foreground = image * mask_3c

    # Background extraction
    background_part = background * (1 - mask_3c)

    # Combine
    result = foreground + background_part

    return result.astype(np.uint8)
