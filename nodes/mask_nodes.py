import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import scipy.ndimage
import numpy as np
from contextlib import nullcontext
import os
import json
# import model_management
 # from nodes import MAX_RESOLUTION

from utility.utility import tensor2pil, pil2tensor

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BatchImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "dilation_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}

    CATEGORY = "MultiCutAndDragWithTruck"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "batch_convert_to_mask"
    DESCRIPTION = "Converts RGB images to binary masks using grayscale conversion and thresholding, with optional morphological dilation"

    def batch_convert_to_mask(self, images, threshold, dilation_amount):
        # Handle case where images is a tuple (common in ComfyUI node system)
        if isinstance(images, tuple):
            images = images[0]

        # Print the shape to diagnose
        print(f"Input images shape: {images.shape}")
        
        # Check the tensor shape and transpose if needed
        if len(images.shape) == 4:  # [B, C, H, W]
            B, C, H, W = images.shape
            if C == 3:
                # Standard RGB format
                pass
            elif W == 3 and C > 3:
                # Dimensions are swapped [B, H, W, C]
                print("Dimensions detected as [B, H, W, C], correcting to [B, C, H, W]")
                # Use correct permutation: from [B, H, W, C] to [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
                B, C, H, W = images.shape
        elif len(images.shape) == 3:  # Could be [B, H, W] or [H, W, C]
            if images.shape[2] == 3:  # [H, W, C] format
                print("Transposing image tensor from [H, W, C] to [1, C, H, W]")
                # Correct permutation: from [H, W, C] to [1, C, H, W]
                images = images.permute(2, 0, 1).unsqueeze(0)
                B, C, H, W = images.shape
            else:
                # Assume [B, H, W] format
                B, H, W = images.shape
                C = 1
        
        print(f"After shape analysis: B={B}, C={C if 'C' in locals() else 1}, H={H}, W={W}")
        
        # Create a properly shaped tensor for masks
        masks = torch.zeros((B, C, H, W), device=images.device)
        
        # Convert each image to a binary mask
        if len(images.shape) == 4:  # [B, C, H, W]
            for i in range(B):
                # Convert RGB to grayscale properly
                if C == 3:
                    # Standard RGB format
                    gray = images[i].mean(dim=0)
                else:
                    # Single channel
                    gray = images[i][0]
                masks[i] = (gray > threshold).float()
        else:  # [B, H, W]
            for i in range(B):
                masks[i] = (images[i] > threshold).float()
        
        # Apply morphological dilation if requested
        if dilation_amount > 0:
            c = 1  # non-tapered corners
            kernel = np.array([[c, 1, c],
                             [1, 1, 1],
                             [c, 1, c]])
            
            dilated_masks = []
            for m in masks:
                # Handle potential multichannel masks
                if len(m.shape) > 2:  # If mask has shape [C, H, W]
                    channels = []
                    for channel in range(m.shape[0]):
                        channel_data = m[channel].cpu().numpy().astype(np.float32)
                        for _ in range(dilation_amount):
                            channel_data = scipy.ndimage.grey_dilation(channel_data, footprint=kernel)
                        channels.append(torch.from_numpy(channel_data))
                    dilated_mask = torch.stack(channels)
                else:  # Single channel [H, W]
                    output = m.cpu().numpy().astype(np.float32)
                    for _ in range(dilation_amount):
                        output = scipy.ndimage.grey_dilation(output, footprint=kernel)
                    dilated_mask = torch.from_numpy(output)
                
                dilated_masks.append(dilated_mask)
            
            masks = torch.stack(dilated_masks).to(images.device)

        print(f"Mask tensor shape: {masks.shape}")
        print(f"Mask value range: {masks.min()} to {masks.max()}")

        return (masks,)

class MapTrajectoriesToSegmentedMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "trajectories": ("STRING", {"forceInput": True}),
                }}

    CATEGORY = "MultiCutAndDragWithTruck"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("MASKS", "TRANSLATED_TRAJECTORIES")
    FUNCTION = "map_trajectories"
    DESCRIPTION = "Maps trajectories to masks by centering the start point of trajectories on mask centroids"

    def map_trajectories(self, masks, trajectories):
        # Parse trajectories JSON
        paths_list = json.loads(trajectories)
        
        # Handle case where masks is a tuple (common in ComfyUI node system)
        if isinstance(masks, tuple):
            masks = masks[0]  # Extract the mask tensor from the tuple
        
        if len(paths_list) != masks.shape[0]:
            raise ValueError(f"Number of trajectories ({len(paths_list)}) must match number of masks ({masks.shape[0]})")
        
        # Get dimensions, handling different tensor formats
        if len(masks.shape) == 4:  # [B, C, H, W]
            B, C, H, W = masks.shape
        elif len(masks.shape) == 3:  # [B, H, W]
            B, H, W = masks.shape
            C = 1
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        
        # Process each mask and trajectory
        translated_paths = []
        
        # Ensure the test directory exists
        # test_dir = os.path.join(script_directory, "test")
        # os.makedirs(test_dir, exist_ok=True)
        
        for i in range(B):
            # Get the mask and convert to numpy
            mask = masks[i]
            mask_np = mask.cpu().numpy()
            
            # Handle masks with different dimensions
            if len(mask_np.shape) > 2:  # Multi-channel mask [C, H, W]
                # Average across channels if needed
                if mask_np.shape[0] > 1:
                    mask_np = mask_np.mean(axis=0)
                else:
                    mask_np = mask_np[0]  # Take first channel
                    
            # Calculate centroid correctly on 2D mask
            if mask_np.sum() > 0:
                # This works on 2D arrays
                indices = np.where(mask_np > 0.5)
                y_indices, x_indices = indices[0], indices[1]
                centroid_y = y_indices.mean()
                centroid_x = x_indices.mean()
            else:
                # Default to center if no mask points
                centroid_x = W / 2
                centroid_y = H / 2
            
            # Debug: Save mask with centroid marked
            # Create a visualization image
            # Ensure mask_np is 2D for visualization
            # if len(mask_np.shape) > 2:
            #     if mask_np.shape[0] > 1:
            #         viz_mask = mask_np.mean(axis=0)
            #     else:
            #         viz_mask = mask_np[0]
            # else:
            #     viz_mask = mask_np
                
            # # Scale to 0-255 for PIL
            # viz_mask_uint8 = (viz_mask * 255).astype(np.uint8)
            
            # # Create RGB image (convert grayscale to RGB)
            # h, w = viz_mask.shape
            # rgb_mask = np.stack([viz_mask_uint8] * 3, axis=2)  # Shape: [H, W, 3]
            
            # # Convert to PIL
            # mask_pil = Image.fromarray(rgb_mask)
            
            # # Create a drawing context
            # draw = ImageDraw.Draw(mask_pil)
            
            # # Draw a red circle at the centroid position (radius=5 pixels)
            # radius = 5
            # draw.ellipse(
            #     (
            #         centroid_x - radius, 
            #         centroid_y - radius, 
            #         centroid_x + radius, 
            #         centroid_y + radius
            #     ), 
            #     fill='red', 
            #     outline='red'
            # )
            
            # # Save the debug image
            # debug_filename = os.path.join(test_dir, f"mask_{i}_centroid.png")
            # mask_pil.save(debug_filename)
            # print(f"Debug mask with centroid saved to: {debug_filename}")
            
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


