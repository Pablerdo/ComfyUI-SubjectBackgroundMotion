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
                "rotations": ("STRING", {"forceInput": True}),
                "scalings": ("STRING", {"forceInput": True}),
                "masks": ("MASK",),
                "frame_width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "bg_image": ("IMAGE",),
                "outpainted_bg_image": ("IMAGE",),
                "num_frames": ("INT", {"forceInput": True}),
            },
            "optional": {
                "camera_motion": ("STRING", {"forceInput": True}),
            }
        }

    def multi_cut_and_drag_with_truck(self, image, coordinate_paths, rotations, scalings, masks, frame_width, frame_height, num_frames, bg_image=None, camera_motion=None, outpainted_bg_image=None):

        if bg_image is None:
            raise ValueError("Background image is required")
                    
        coordinate_paths_list = json.loads(coordinate_paths)
        rotation_list = json.loads(rotations)
        scalings_list = json.loads(scalings)

        # # Verify that the cooridnate paths, rotation and scaling lists are the same length as the number of frames
        # if len(coordinate_paths_list) != num_frames:
        #     raise ValueError(f"Number of coordinate paths ({len(coordinate_paths_list)}) must match number of frames ({num_frames})")

        # if len(rotation_list) != num_frames:
        #     raise ValueError(f"Number of rotations ({len(rotation_list)}) must match number of frames ({num_frames})")

        # if len(scalings_list) != num_frames:
        #     raise ValueError(f"Number of scalings ({len(scalings_list)}) must match number of frames ({num_frames})")

        camera_motion = json.loads(camera_motion)
        
        truck_vector = camera_motion["truck_vector"]

        # Handle case where masks is a tuple (common in ComfyUI node system)
        masks_tensor = masks
        if isinstance(masks, tuple):
            masks_tensor = masks[0]  # Extract the mask tensor from the tuple
            
        return self.create_animation(image, coordinate_paths_list, rotation_list, scalings_list, masks, frame_width, frame_height,  num_frames, bg_image, truck_vector, outpainted_bg_image)

    def create_animation(self, image, paths_list, rotations_list, scalings_list, masks, frame_width, frame_height, num_frames, bg_image=None, truck_vector=None, outpainted_bg_image=None):
        
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

        background = tensor2pil(bg_image)[0]

        outpainted_background = tensor2pil(outpainted_bg_image)[0]

        truck_trajectory, camera_vector_list, adjusted_truck_vector = self._calculate_truck_trajectory(truck_vector, background.size[0], background.size[1], num_frames)

        adjusted_subject_trajectories = self._calculate_subject_trajectories_with_truck_vector(paths_list, adjusted_truck_vector, num_frames)

        outpaint_x_margin = (outpainted_background.size[0] - background.size[0]) / 2
        outpaint_y_margin = (outpainted_background.size[1] - background.size[1]) / 2

        # Create a new background images that is the same size as the input image
        background_images = [None] * num_frames
        for frame_idx in range(num_frames):
            # Create a new image with the background shifted by the truck trajectory
            result_background = background.copy()
            # translated_background = background.copy()
            moving_background = outpainted_background.copy()
            result_background.paste(moving_background, (int(truck_trajectory[frame_idx]["x"] - outpaint_x_margin), int(truck_trajectory[frame_idx]["y"] - outpaint_y_margin)))

            background_images[frame_idx] = result_background
        
        # Print the adjusted subject trajectories

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Adjusted subject trajectories:")
        print(adjusted_subject_trajectories)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # Print the truck trajectory
        print("Truck trajectory:")
        print(truck_trajectory)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

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
                    'coords': adjusted_subject_trajectories[mask_idx],
                    'rotations': rotations_list[mask_idx],
                    'scalings': scalings_list[mask_idx],
                    'crop_offset_x': x_min,
                    'crop_offset_y': y_min
                })

        # Create batch of images with cut regions at different positions
        for frame_idx in range(num_frames):
            # Get the background image for this frame
            # new_image = background.copy()
            new_image = background_images[frame_idx].copy()
            new_mask = Image.new("L", (frame_width, frame_height), 0)

            # Place each cut region at its position for this frame
            for region in cut_regions:
                # Get rotation and scaling values for this frame
                rotation_angle = region['rotations'][frame_idx]
                scale_factor = region['scalings'][frame_idx]
                
                # Start with the original image and mask
                current_image = region['image'].copy()
                current_mask = region['mask'].copy()
                
                # The trajectory coordinates are where we want the centroid to be in the final output
                trajectory_x = region['coords'][frame_idx]['x']
                trajectory_y = region['coords'][frame_idx]['y']
                
                # Calculate the actual centroid position within the cropped region
                # This is the center of mass of the white pixels in the mask
                mask_array = np.array(current_mask)
                y_indices, x_indices = np.where(mask_array > 128)
                
                if len(x_indices) > 0 and len(y_indices) > 0:
                    # Calculate centroid within the cropped image
                    centroid_x = x_indices.mean()
                    centroid_y = y_indices.mean()
                else:
                    # Fallback to center of cropped image if no mask pixels
                    centroid_x = current_image.width / 2
                    centroid_y = current_image.height / 2
                
                # Create a large working canvas that can accommodate any transformation
                # Make it large enough to handle scaling up and rotation without clipping
                max_dimension = max(current_image.width, current_image.height)
                working_canvas_size = int(max_dimension * max(scale_factor, 1.0) * 4)  # Very large canvas
                
                # Create working canvas
                working_image = Image.new("RGBA", (working_canvas_size, working_canvas_size), (0, 0, 0, 0))
                working_mask = Image.new("L", (working_canvas_size, working_canvas_size), 0)
                
                # Place the original image/mask so that the centroid is at the canvas center
                canvas_center = working_canvas_size // 2
                paste_x = canvas_center - int(centroid_x)
                paste_y = canvas_center - int(centroid_y)
                
                working_image.paste(current_image, (paste_x, paste_y))
                working_mask.paste(current_mask, (paste_x, paste_y))
                
                # Apply scaling around the canvas center (which is now the centroid)
                if scale_factor != 1.0:
                    scaled_size = int(working_canvas_size * scale_factor)
                    working_image = working_image.resize((scaled_size, scaled_size), Image.LANCZOS)
                    working_mask = working_mask.resize((scaled_size, scaled_size), Image.LANCZOS)
                    
                    # Update canvas center after scaling
                    canvas_center = scaled_size // 2
                    working_canvas_size = scaled_size
                
                # Apply rotation around the canvas center (which is still the centroid)
                if rotation_angle != 0:
                    working_image = working_image.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
                    working_mask = working_mask.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
                
                # Now we have the fully transformed image/mask with centroid at canvas center
                # Calculate where to place this on the final output so the centroid lands on trajectory point
                final_paste_x = int(trajectory_x - canvas_center)
                final_paste_y = int(trajectory_y - canvas_center)
                
                # Paste the entire transformed result onto the output
                # No cropping - let it extend beyond if needed
                new_image.paste(working_image, (final_paste_x, final_paste_y), working_mask)
                new_mask.paste(working_mask, (final_paste_x, final_paste_y))

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
        for i in range(num_frames + 1):
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

        return trajectory, camera_vector_list, adjusted_truck_vector
    
    def _calculate_subject_trajectories_with_truck_vector(self, subject_trajectories, adjusted_truck_vector, num_frames):
        # This function will calculate the trajectory of the subject, given the camera movement.
        # The camera movement is dealt with by receiving a camera_movement_
        # The trajectory vector is the amount that the subject will move in each frame, the camera vector is the amount that the camera will move in each frame. 
        # Thus, we need to sum them in order to get the total movement of the subject, in the context of the video window

        # Convert the subject trajectories to a list of vectors here
        import utility.vector_utilities as vector_utilities

        subject_vectors_list = vector_utilities.trajectory_list_to_vector_list(subject_trajectories)

        adjusted_subject_trajectories = []

        frame_relevant_adjusted_truck_vector = {"x": adjusted_truck_vector["x"] * (1 / num_frames), "y": adjusted_truck_vector["y"] * (1 / num_frames)}

       # Process each subject trajectory separately.
        for subject_traj, subject_vectors in zip(subject_trajectories, subject_vectors_list):
            adjusted_traj = []
            # Use the first coordinate as the starting position.
            current_position = subject_traj[0]
            adjusted_traj.append(current_position)
            
            # For each frame (except the last), compute the new position.
            for i in range(num_frames - 1):
                subj_vector = subject_vectors[i]
                # Sum the subject's movement and the camera movement.
                adjusted_vector = vector_utilities.add_vectors(subj_vector, frame_relevant_adjusted_truck_vector)
                # Update the current position using the adjusted movement.
                current_position = {
                    "x": current_position["x"] + adjusted_vector["x"],
                    "y": current_position["y"] + adjusted_vector["y"]
                }
                adjusted_traj.append(current_position)
            
            adjusted_subject_trajectories.append(adjusted_traj)
        
        return adjusted_subject_trajectories
        

    