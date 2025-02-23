import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import scipy.ndimage
import numpy as np
from contextlib import nullcontext
import os

import model_management
from comfy.utils import ProgressBar
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION

import folder_paths

from ..utility.utility import tensor2pil, pil2tensor

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BatchImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "dilation_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}

    CATEGORY = "PSNodes/masking"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "batch_convert_to_mask"
    DESCRIPTION = "Converts transparent PNG images to binary masks with optional dilation"

    def batch_convert_to_mask(self, images, dilation_amount):
        B, H, W, C = images.shape
        masks = []

        # Convert images to masks based on alpha channel
        for i in range(B):
            # Extract alpha channel (assuming RGBA where A is the last channel)
            mask = images[i, :, :, -1]  # Get alpha channel
            mask = (mask > 0.5).float()  # Binarize the mask
            masks.append(mask)

        mask_tensor = torch.stack(masks)

        # Apply morphological dilation if requested
        if dilation_amount > 0:
            c = 1  # non-tapered corners
            kernel = np.array([[c, 1, c],
                             [1, 1, 1],
                             [c, 1, c]])
            
            dilated_masks = []
            for m in mask_tensor:
                output = m.cpu().numpy().astype(np.float32)
                for _ in range(dilation_amount):
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
                dilated_masks.append(torch.from_numpy(output))
            
            mask_tensor = torch.stack(dilated_masks)

        return (mask_tensor,)