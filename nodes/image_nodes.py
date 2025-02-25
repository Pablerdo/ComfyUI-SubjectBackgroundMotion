import sys
import os

import cv2
from PIL import Image, ImageOps
# import folder_paths
import torch
import numpy as np
from io import BytesIO
import base64

# Credit to https://github.com/glowcone

class LoadImageFromBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "PSNodes/image"

    def convert_color(self, image):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def load_image(self, data):
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)

        result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        channels = cv2.split(result)
        if len(channels) > 3:
            mask = channels[3].astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(channels[0].shape, dtype=torch.float32, device="cpu")

        result = self.convert_color(result)
        result = result.astype(np.float32) / 255.0
        image = torch.from_numpy(result)[None,]
        return image, mask.unsqueeze(0)


class LoadImagesFromBase64Array:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_array": ("STRING", {"default": "[]"}),  # Expects a JSON string array of base64 encodings
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_images"
    CATEGORY = "PSNodes/image"

    def convert_color(self, image):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_images(self, data_array):
        import json
        
        # Parse the JSON string array
        try:
            base64_array = json.loads(data_array)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON array string provided")
        
        if not isinstance(base64_array, list):
            raise ValueError("Input must be a JSON array string")
            
        images = []
        masks = []
        
        # Process each base64 string
        for data in base64_array:
            nparr = np.frombuffer(base64.b64decode(data), np.uint8)
            result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            channels = cv2.split(result)
            if len(channels) > 3:
                mask = channels[3].astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
            else:
                mask = torch.ones(channels[0].shape, dtype=torch.float32, device="cpu")
            
            result = self.convert_color(result)
            result = result.astype(np.float32) / 255.0
            image = torch.from_numpy(result)[None,]
            
            images.append(image)
            masks.append(mask.unsqueeze(0))
        
        # Stack all images and masks
        batched_images = torch.cat(images, dim=0)
        batched_masks = torch.cat(masks, dim=0)
        
        return (batched_images, batched_masks)

NODE_CLASS_MAPPINGS = {
    "LoadImageFromBase64": LoadImageFromBase64,
    "LoadImagesFromBase64Array": LoadImagesFromBase64Array,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromBase64": "Load Image From Base64",
    "LoadImagesFromBase64Array": "Load Images From Base64 Array",
}