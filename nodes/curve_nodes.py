import torch
from torchvision import transforms
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter, ImageChops
import numpy as np
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.utility import pil2tensor, tensor2pil
import io
import base64
        

class MultiCutAndDragOnPath:
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image","mask", )
    FUNCTION = "multi_cutanddrag" 
    CATEGORY = "PSNodes/experimental"
    DESCRIPTION = """
    Cut and drag parts of an image along specified coordinate paths.
    Coordinate paths should be an array of arrays containing coordinate objects, e.g.:
    [
        [{"x": 400, "y": 240}, {"x": 720, "y": 480}],  # First path
        [{"x": 720, "y": 480}, {"x": 400, "y": 240}]   # Second path
    ]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "coordinate_paths": ("STRING", {"forceInput": True}),
                "masks": ("MASK",),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "inpaint": ("BOOLEAN", {"default": True}),
                "mode": (["translate", "translate_and_rotate"], {"default": "translate"}),
            },
            "optional": {
                "bg_image": ("IMAGE",),
                "degrees": ("STRING", {"forceInput": True}),
            }
        }

    def multi_cutanddrag(self, image, coordinate_paths, masks, frame_width, frame_height, inpaint, rotation=False, bg_image=None, degrees=[0.0]):
        # Handle case where masks is a tuple (common in ComfyUI node system)
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]  # Extract the mask tensor from the tuple
            
        if not rotation:
            return self._translate(image, coordinate_paths, masks, frame_width, frame_height, inpaint, bg_image)
        else:
            # Verify that degrees array matches number of masks
            if len(degrees) != masks_tensor.shape[0]:
                raise ValueError(f"Number of rotation degrees ({len(degrees)}) must match number of masks ({masks_tensor.shape[0]})")
            return self._translate_and_rotate(image, coordinate_paths, masks, frame_width, frame_height, inpaint, degrees, bg_image)

    def _translate(self, image, coordinate_paths, masks, frame_width, frame_height, inpaint, bg_image=None):
        # Parse coordinate paths as array of arrays
        paths_list = json.loads(coordinate_paths)
        
        # Handle case where masks is a tuple
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]
            
        if len(paths_list) != masks_tensor.shape[0]:
            raise ValueError(f"Number of coordinate paths ({len(paths_list)}) must match number of masks ({masks_tensor.shape[0]})")

        batch_size = len(paths_list[0])  # Number of frames to generate
        images_list = []
        masks_list = []

        # Convert input image to PIL
        input_image = tensor2pil(image[0])[0]
        
        # Create inpainted background once if needed
        if bg_image is None:
            background = input_image.copy()
            if inpaint:
                import cv2
                # Create combined mask for all cut areas
                combined_mask = Image.new("L", background.size, 0)
                draw = ImageDraw.Draw(combined_mask)
                border = 5
                
                for mask_idx in range(masks_tensor.shape[0]):
                    # Get mask and ensure it's properly shaped for PIL conversion
                    mask_tensor = masks_tensor[mask_idx]
                    
                    # Handle potentially problematic dimensions
                    if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:
                        mask_tensor = mask_tensor.squeeze(0)  # Remove singleton dimension
                    
                    # Convert to numpy array directly instead of using tensor2pil
                    mask_array = mask_tensor.cpu().numpy()
                    
                    # Ensure mask_array is 2D
                    if len(mask_array.shape) == 3:
                        if mask_array.shape[0] == 1:
                            mask_array = mask_array[0]  # Take first channel
                        else:
                            mask_array = mask_array.mean(axis=0)  # Average channels
                    
                    y_indices, x_indices = np.where(mask_array > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        draw.rectangle([x_min-border, y_min-border, x_max+border, y_max+border], fill=255)
                
                background = cv2.inpaint(
                    np.array(background), 
                    np.array(combined_mask), 
                    inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA
                )
                background = Image.fromarray(background)
        else:
            background = tensor2pil(bg_image)[0]

        # Cut out each masked region and store info
        cut_regions = []
        for mask_idx in range(masks_tensor.shape[0]):
            # Get mask and ensure it's properly shaped
            mask_tensor = masks_tensor[mask_idx]
            
            # Handle potentially problematic dimensions
            if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)  # Remove singleton dimension
            
            # Convert to numpy array directly
            mask_array = mask_tensor.cpu().numpy()
            
            # Ensure mask_array is 2D
            if len(mask_array.shape) == 3:
                if mask_array.shape[0] == 1:
                    mask_array = mask_array[0]  # Take first channel
                else:
                    mask_array = mask_array.mean(axis=0)  # Average channels
            
            # Create PIL mask directly from the array
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
            
            # Rest of the processing remains the same
            y_indices, x_indices = np.where(mask_array > 0)
            
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                cut_width = x_max - x_min
                cut_height = y_max - y_min
                cut_image = input_image.crop((x_min, y_min, x_max, y_max))
                cut_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
                cut_regions.append({
                    'image': cut_image,
                    'mask': cut_mask,
                    'width': cut_width,
                    'height': cut_height,
                    'coords': paths_list[mask_idx]
                })

        # Create batch of images with cut regions at different positions
        for frame_idx in range(batch_size):
            new_image = background.copy()
            new_mask = Image.new("L", (frame_width, frame_height), 0)

            # Place each cut region at its position for this frame
            for region in cut_regions:
                target_x = int(region['coords'][frame_idx]['x'] - region['width']/2)
                target_y = int(region['coords'][frame_idx]['y'] - region['height']/2)
                
                new_image.paste(region['image'], (target_x, target_y), region['mask'])
                new_mask.paste(region['mask'], (target_x, target_y))

            # Convert to tensor and append
            image_tensor = pil2tensor(new_image)
            mask_tensor = pil2tensor(new_mask)
            
            images_list.append(image_tensor)
            masks_list.append(mask_tensor)

        # Stack tensors into batches
        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)

        return (out_images, out_masks)

    def _translate_and_rotate(self, image, coordinate_paths, masks, frame_width, frame_height, inpaint, degrees, bg_image=None):
        # Parse coordinate paths as array of arrays
        paths_list = json.loads(coordinate_paths)
        
        # Convert degrees string to float array
        try:
            degrees_list = json.loads(degrees)
            if isinstance(degrees_list, (int, float)):
                degrees_list = [float(degrees_list)]
            else:
                degrees_list = [float(d) for d in degrees_list]
        except json.JSONDecodeError:
            # If not valid JSON, try to convert single string to float
            try:
                degrees_list = [float(degrees)]
            except ValueError:
                raise ValueError("Degrees must be a valid number or JSON array of numbers")

        # Handle case where masks is a tuple
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]

        if len(paths_list) != masks_tensor.shape[0]:
            raise ValueError(f"Number of coordinate paths ({len(paths_list)}) must match number of masks ({masks_tensor.shape[0]})")

        # Verify that degrees array matches number of masks
        if len(degrees_list) != masks_tensor.shape[0]:
            raise ValueError(f"Number of rotation degrees ({len(degrees_list)}) must match number of masks ({masks_tensor.shape[0]})")

        batch_size = len(paths_list[0])  # Number of frames to generate
        images_list = []
        masks_list = []

        # Convert input image to PIL
        input_image = tensor2pil(image)[0]
        
        # Create inpainted background once if needed
        if bg_image is None:
            background = input_image.copy()
            if inpaint:
                import cv2
                # Create combined mask for all cut areas
                combined_mask = Image.new("L", background.size, 0)
                draw = ImageDraw.Draw(combined_mask)
                border = 5
                
                for mask_idx in range(masks_tensor.shape[0]):
                    mask_pil = tensor2pil(masks_tensor[mask_idx:mask_idx+1])[0]
                    mask_array = np.array(mask_pil)
                    y_indices, x_indices = np.where(mask_array > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        draw.rectangle([x_min-border, y_min-border, x_max+border, y_max+border], fill=255)
                
                background = cv2.inpaint(
                    np.array(background), 
                    np.array(combined_mask), 
                    inpaintRadius=3, 
                    flags=cv2.INPAINT_TELEA
                )
                background = Image.fromarray(background)
        else:
            background = tensor2pil(bg_image)[0]

        # Cut out each masked region and store info
        cut_regions = []
        for mask_idx in range(masks_tensor.shape[0]):
            mask_pil = tensor2pil(masks_tensor[mask_idx:mask_idx+1])[0]
            mask_array = np.array(mask_pil)
            y_indices, x_indices = np.where(mask_array > 0)
            
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                cut_width = x_max - x_min
                cut_height = y_max - y_min
                cut_image = input_image.crop((x_min, y_min, x_max, y_max))
                cut_mask = mask_pil.crop((x_min, y_min, x_max, y_max))
                cut_regions.append({
                    'image': cut_image,
                    'mask': cut_mask,
                    'width': cut_width,
                    'height': cut_height,
                    'coords': paths_list[mask_idx],
                    'degrees': degrees_list[mask_idx]  # Store the rotation amount for this region
                })

        # Create batch of images with cut regions at different positions
        for frame_idx in range(batch_size):
            new_image = background.copy()
            new_mask = Image.new("L", (frame_width, frame_height), 0)

            # Place each cut region at its position for this frame
            for region in cut_regions:
                # Calculate rotation angle for this frame using the region's specific degrees
                rotation_angle = (region['degrees'] * frame_idx) / (batch_size - 1)
                
                # Rotate the cut region and its mask
                rotated_image = region['image'].rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
                rotated_mask = region['mask'].rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
                
                # Get the new dimensions after rotation
                rotated_width, rotated_height = rotated_image.size
                
                # Calculate the position accounting for the new dimensions
                target_x = int(region['coords'][frame_idx]['x'] - rotated_width/2)
                target_y = int(region['coords'][frame_idx]['y'] - rotated_height/2)
                
                # Paste the rotated image and mask
                new_image.paste(rotated_image, (target_x, target_y), rotated_mask)
                new_mask.paste(rotated_mask, (target_x, target_y))

            # Convert to tensor and append
            image_tensor = pil2tensor(new_image)
            mask_tensor = pil2tensor(new_mask)
            
            images_list.append(image_tensor)
            masks_list.append(mask_tensor)

        # Stack tensors into batches
        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)

        return (out_images, out_masks)