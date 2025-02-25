from .nodes.curve_nodes import MultiCutAndDragOnPath
from .nodes.mask_nodes import BatchImageToMask
from .nodes.image_nodes import LoadImagesFromBase64Array
from .nodes.image_nodes import LoadImageFromBase64
from .nodes.mask_nodes import MapTrajectoriesToSegmentedMasks

NODE_CONFIG = {
    "LoadImageFromBase64": {"class": LoadImageFromBase64, "name": "LoadImageFromBase64"},
    "LoadImagesFromBase64Array": {"class": LoadImagesFromBase64Array, "name": "LoadImagesFromBase64Array"},
    "MultiCutAndDragOnPath": {"class": MultiCutAndDragOnPath, "name": "MultiCutAndDragOnPath"},
    "BatchImageToMask": {"class": BatchImageToMask, "name": "BatchImageToMask"},
    "MapTrajectoriesToSegmentedMasks": {"class": MapTrajectoriesToSegmentedMasks, "name": "MapTrajectoriesToSegmentedMasks"}

}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
