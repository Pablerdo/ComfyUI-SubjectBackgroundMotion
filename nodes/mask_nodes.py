import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import scipy.ndimage
import numpy as np
from contextlib import nullcontext
import os
import json
import model_management
from nodes import MAX_RESOLUTION

from ..utility.utility import tensor2pil, pil2tensor

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BatchImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "dilation_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}

    CATEGORY = "PSNodes/masking"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "batch_convert_to_mask"
    DESCRIPTION = "Converts RGB images to binary masks using grayscale conversion and thresholding, with optional morphological dilation"

    def batch_convert_to_mask(self, images, threshold, dilation_amount):
        B, H, W, C = images.shape
        masks_tensor = torch.zeros((B, H, W), device=images.device)
        
        # Convert each RGB image to a binary mask
        for i in range(B):
            img = images[i]
            gray = img.mean(dim=-1)  # Simple RGB to grayscale conversion
            mask = (gray > threshold).float()  # Binarize using threshold
            masks_tensor[i] = mask
            
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

        return (masks_tensor,)

class MapTrajectoriesToSegmentedMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "trajectories": ("STRING", {"forceInput": True}),
                }}

    CATEGORY = "PSNodes/masking"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("MASKS", "TRANSLATED_TRAJECTORIES")
    FUNCTION = "map_trajectories"
    DESCRIPTION = "Maps trajectories to masks by centering the start point of trajectories on mask centroids"

    def map_trajectories(self, masks, trajectories):
        # Parse trajectories JSON
        paths_list = json.loads(trajectories)
        
        if len(paths_list) != masks.shape[0]:
            raise ValueError(f"Number of trajectories ({len(paths_list)}) must match number of masks ({masks.shape[0]})")
        
        B, H, W = masks.shape
        
        # Process each mask and trajectory
        translated_paths = []
        
        for i in range(B):
            # Get the mask
            mask = masks[i]
            
            # Calculate centroid
            mask_np = mask.cpu().numpy()
            if mask_np.sum() > 0:
                y_indices, x_indices = np.where(mask_np > 0.5)
                centroid_x = x_indices.mean()
                centroid_y = y_indices.mean()
            else:
                # Default to center if no mask points
                centroid_x = W / 2
                centroid_y = H / 2
            
            # Get corresponding trajectory and translate it
            path = paths_list[i]
            
            # Calculate the offset from the first point to the centroid
            if len(path) > 0:
                first_point = path[0]
                offset_x = centroid_x - first_point["x"]
                offset_y = centroid_y - first_point["y"]
                
                # Apply the offset to all points in the path
                translated_path = [{"x": point["x"] + offset_x, "y": point["y"] + offset_y} for point in path]
                translated_paths.append(translated_path)
            else:
                # Empty path case
                translated_paths.append([])
        
        # Convert translated paths back to JSON
        translated_trajectories_json = json.dumps(translated_paths)
        
        return (masks, translated_trajectories_json)

