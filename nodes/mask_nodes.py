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

from utility.utility import tensor2pil, pil2tensor

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_RESOLUTION=16384

class BatchImageToMask:
    CATEGORY = "SubjectBackgroundMotion"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "batch_convert_to_mask"
    DESCRIPTION = "Converts RGB images to binary masks using grayscale conversion and thresholding, with optional morphological dilation"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "dilation_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}


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

    CATEGORY = "SubjectBackgroundMotion"
    RETURN_TYPES = ("MASK", "STRING",)
    RETURN_NAMES = ("masks", "translated_trajectories",)
    FUNCTION = "map_trajectories"
    DESCRIPTION = "Maps trajectories to masks by centering the start point of trajectories on mask centroids"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "trajectories": ("STRING", {"forceInput": True}),
                }}

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


class PadAndTranslateImageForOutpainting:
    CATEGORY = "SubjectBackgroundMotion"
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("final_background_placement", "placement_mask",)
    FUNCTION = "pad_and_translate_image_for_outpainting"
    DESCRIPTION = "Pads and translates an image for outpainting"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "bg_image": ("IMAGE",),
                    "truck_vector": ("STRING", {"forceInput": True}),
                    "frame_width": ("INT", {"forceInput": True}),
                    "frame_height": ("INT", {"forceInput": True}),
                    "feathering": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}
        
    def pad_and_translate_image_for_outpainting(self, bg_image, truck_vector, frame_width, frame_height, feathering=0):
        # Parse the truck vector
        truck_vector = json.loads(truck_vector)

        # Calculate pixel offset from the truck vector
        x_offset = int((0.25 * frame_width) * truck_vector["x"])
        y_offset = int((0.25 * frame_height) * truck_vector["y"])
        
        # Get the dimensions of the background image
        if isinstance(bg_image, tuple):
            bg_image = bg_image[0]
        
        pil_bg_image = tensor2pil(bg_image[0])[0]
        
        pil_bg_image_width, pil_bg_image_height = pil_bg_image.width, pil_bg_image.height

        # Create output tensors
        translated_images = []
        masks = []

        # Create a gray background image
        # gray_background = Image.new("RGB", (pil_bg_image_width, pil_bg_image_height), color=(128, 128, 128))
        
        # making a black background image to experiment
        gray_background = Image.new("RGB", (pil_bg_image_width, pil_bg_image_height), color=(0, 0, 0))

        # Create a mask image white initially
        mask_image = Image.new("L", (pil_bg_image_width, pil_bg_image_height), color=255)
        
        # Calculate paste position
        paste_x = max(0, x_offset)
        paste_y = max(0, y_offset)
        
        # Calculate how much of the original image to use
        crop_left = max(0, -x_offset)
        crop_top = max(0, -y_offset)
        crop_right = min(pil_bg_image_width, pil_bg_image_width - x_offset if x_offset < 0 else pil_bg_image_width)
        crop_bottom = min(pil_bg_image_height, pil_bg_image_height - y_offset if y_offset < 0 else pil_bg_image_height)
        
        # Crop the original image if needed
        if crop_left > 0 or crop_top > 0 or crop_right < pil_bg_image_width or crop_bottom < pil_bg_image_height:
            cropped_image = pil_bg_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        else:
            cropped_image = pil_bg_image
        
        # Create a mask for the pasted region
        paste_mask = Image.new("L", cropped_image.size, color=0)
        
        # Paste the image onto the gray background
        gray_background.paste(cropped_image, (paste_x, paste_y), None)
        
        # Paste the mask
        mask_image.paste(paste_mask, (paste_x, paste_y), None)
        
        # Apply feathering if requested
        if feathering > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=feathering/3))
        
        # Convert back to tensor
        translated_image = pil2tensor(gray_background)
        mask_tensor = pil2tensor(mask_image)
        
        masks.append(mask_tensor)
        
        # # Stack results
        # out_images = torch.cat(translated_images, dim=0)
        out_masks = torch.cat(masks, dim=0)
        
        return (translated_image, out_masks)


        
class PadForOutpaintGeneral:

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "expand_image"
    CATEGORY = "SubjectBackgroundMotion"

    # credits to ComfyAnon but I need one that had black background
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.zeros(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        )

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)

        
        
