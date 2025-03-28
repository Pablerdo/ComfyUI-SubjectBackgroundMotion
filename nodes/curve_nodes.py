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

class MultiCutAndDragWithTruck:
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image","mask",)
    FUNCTION = "multi_cut_and_drag_with_truck"
    CATEGORY = "SubjectBackgroundMotion"
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
                # "rotation": ("BOOLEAN", {"default": False}),
                "bg_image": ("IMAGE",),
                "num_frames": ("INT", {"forceInput": True}),
            },
            "optional": {
                "truck_vector": ("STRING", {"forceInput": True}),
                # "degrees": ("STRING", {"forceInput": True}),
            }
        }

    def multi_cut_and_drag_with_truck(self, image, coordinate_paths, masks, frame_width, frame_height, bg_image=None, truck_vector=None, num_frames=49): # rotation=False, bg_image=None, truck_vector=None, num_frames=49, degrees=[0.0]):

        if bg_image is None:
            raise ValueError("Background image is required")
                    
        coordinate_paths_list = json.loads(coordinate_paths)

        truck_vector = json.loads(truck_vector)

        # if len(coordinate_paths_list) != num_frames:
        #     raise ValueError(f"Number of coordinate paths ({len(coordinate_paths_list)}) must match number of frames ({num_frames})")
        
        # Handle case where masks is a tuple (common in ComfyUI node system)
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]  # Extract the mask tensor from the tuple
            
        return self._translate_with_truck(image, coordinate_paths_list, masks, frame_width, frame_height, bg_image, truck_vector, num_frames)
    
        # if not rotation:
        #     return self._translate_with_truck(image, coordinate_paths_list, masks, frame_width, frame_height, inpaint, bg_image, truck_vector, num_frames)
        # else:
        #     # Verify that degrees array matches number of masks
        #     if len(degrees) != masks_tensor.shape[0]:
        #         raise ValueError(f"Number of rotation degrees ({len(degrees)}) must match number of masks ({masks_tensor.shape[0]})")
        #     return self._translate_and_rotate(image, coordinate_paths, masks, frame_width, frame_height, inpaint, degrees, bg_image)

    def _translate_with_truck(self, image, paths_list, masks, frame_width, frame_height, bg_image=None, truck_vector=None, num_frames=50):
        # Parse coordinate paths as array of arrays
        
        # Handle case where masks is a tuple
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]
            
        if len(paths_list) != masks_tensor.shape[0]:
            raise ValueError(f"Number of coordinate paths ({len(paths_list)}) must match number of masks ({masks_tensor.shape[0]})")
        
        images_list = []
        masks_list = []

        # Convert input image to PIL
        input_image = tensor2pil(image[0])[0]
        
        # Create inpainted background once if needed
        # if bg_image is None:
        #     background = input_image.copy()
        #     if inpaint:
        #         import cv2
        #         # Create combined mask for all cut areas
        #         combined_mask = Image.new("L", background.size, 0)
        #         draw = ImageDraw.Draw(combined_mask)
        #         border = 5
                
        #         for mask_idx in range(masks_tensor.shape[0]):
        #             # Get mask and ensure it's properly shaped for PIL conversion
        #             mask_tensor = masks_tensor[mask_idx]
                    
        #             # Handle potentially problematic dimensions
        #             if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:
        #                 mask_tensor = mask_tensor.squeeze(0)  # Remove singleton dimension
                    
        #             # Convert to numpy array directly instead of using tensor2pil
        #             mask_array = mask_tensor.cpu().numpy()
                    
        #             # Ensure mask_array is 2D
        #             if len(mask_array.shape) == 3:
        #                 if mask_array.shape[0] == 1:
        #                     mask_array = mask_array[0]  # Take first channel
        #                 else:
        #                     mask_array = mask_array.mean(axis=0)  # Average channels
                    
        #             y_indices, x_indices = np.where(mask_array > 0)
        #             if len(x_indices) > 0 and len(y_indices) > 0:
        #                 x_min, x_max = x_indices.min(), x_indices.max()
        #                 y_min, y_max = y_indices.min(), y_indices.max()
        #                 draw.rectangle([x_min-border, y_min-border, x_max+border, y_max+border], fill=255)
                
        #         background = cv2.inpaint(
        #             np.array(background), 
        #             np.array(combined_mask), 
        #             inpaintRadius=3, 
        #             flags=cv2.INPAINT_TELEA
        #         )
        #         background = Image.fromarray(background)
        # else:

        background = tensor2pil(bg_image)[0]

        truck_trajectory, adjusted_truck_vector = self._calculate_truck_trajectory(truck_vector, background.size[0], background.size[1], num_frames)

        adjusted_subject_trajectories = self._calculate_subject_trajectories_with_camera_movement(paths_list, adjusted_truck_vector, num_frames)

        # Create a new back of background images that is the same size as the input image
        background_images = [None] * num_frames
        for frame_idx in range(num_frames):
            # Create a new image with the background shifted by the truck trajectory
            stable_background = background.copy()
            translated_background = background.copy()
            stable_background.paste(translated_background, (int(truck_trajectory[frame_idx]["x"]), int(truck_trajectory[frame_idx]["y"])))

            background_images[frame_idx] = stable_background

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
                    'coords': adjusted_subject_trajectories[mask_idx]
                })

        # Create batch of images with cut regions at different positions
        for frame_idx in range(num_frames):
            # Get the background image for this frame
            # new_image = background.copy()
            new_image = background_images[frame_idx].copy()
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

    def _translate_and_rotate(self, image, coordinate_paths, masks, frame_width, frame_height, inpaint, degrees, bg_image=None, truck_vector=None, num_frames=49):
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
        for frame_idx in range(num_frames):
            new_image = background.copy()
            new_mask = Image.new("L", (frame_width, frame_height), 0)

            # Place each cut region at its position for this frame
            for region in cut_regions:
                # Calculate rotation angle for this frame using the region's specific degrees
                rotation_angle = (region['degrees'] * frame_idx) / (num_frames)
                
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

    def _calculate_truck_trajectory(self, truck_vector, frame_width, frame_height, num_frames):
        # The truck trajectory will be the path of the top left of the original, inpainted background.

        # Parse the truck vector
        
        # Extract the start and end points, the adjusted_tuck_vector is the full, raw distance that the bg_image will be moved by
        adjusted_truck_vector = {"x": ((0.25 * frame_width) * truck_vector["x"]), "y": ((0.25 * frame_height) * truck_vector["y"])}
        
        # Calculate the trajectory, we want to go from the start point to the end point in num_frames steps. We are starting from the center of the image, 
        # so the start point is the center of the image.

        # The end point is the middle of the image plus the adjusted truck vector.
        trajectory = []
        for i in range(50):
            trajectory.append({
                "x": 0 + (adjusted_truck_vector["x"]) * i/num_frames,
                "y": 0 + (adjusted_truck_vector["y"]) * i/num_frames
            })

        camera_vector_list = []
        for i in range(num_frames):
            camera_vector_list.append({
                "x": adjusted_truck_vector["x"] * i/num_frames,
                "y": adjusted_truck_vector["y"] * i/num_frames,
            })

        return trajectory, camera_vector_list
    
    def _calculate_subject_trajectories_with_camera_movement(self, subject_trajectories, camera_vector_list, num_frames):
        # This function will calculate the trajectory of the subject, given the camera movement.
        # The camera movement is dealt with by receiving a camera_movement_
        # The trajectory vector is the amount that the subject will move in each frame, the camera vector is the amount that the camera will move in each frame. 
        # Thus, we need to sum them in order to get the total movement of the subject, in the context of the video window

        # Convert the subject trajectories to a list of vectors here
        import utility.vector_utilities as vector_utilities

        subject_vectors_list = vector_utilities.trajectory_list_to_vector_list(subject_trajectories)

        adjusted_subject_trajectories = []

       # Process each subject trajectory separately.
        for subject_traj, subject_vectors in zip(subject_trajectories, subject_vectors_list):
            adjusted_traj = []
            # Use the first coordinate as the starting position.
            current_position = subject_traj[0]
            adjusted_traj.append(current_position)
            
            # For each frame (except the first), compute the new position.
            for i in range(num_frames - 1):
                subj_vector = subject_vectors[i]
                cam_vector = camera_vector_list[i]
                # Sum the subject's movement and the camera movement.
                adjusted_vector = vector_utilities.decrease_vectors(subj_vector, cam_vector)
                # Update the current position using the adjusted movement.
                current_position = {
                    "x": current_position["x"] + adjusted_vector["x"],
                    "y": current_position["y"] + adjusted_vector["y"]
                }
                adjusted_traj.append(current_position)
            
            adjusted_subject_trajectories.append(adjusted_traj)
        
        return adjusted_subject_trajectories
        

    