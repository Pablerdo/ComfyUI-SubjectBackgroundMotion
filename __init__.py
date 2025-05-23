from .nodes.curve_nodes import MultiCutAndDragWithTruck
from .nodes.mask_nodes import BatchImageToMask
from .nodes.mask_nodes import MapTrajectoriesToSegmentedMasks
from .nodes.mask_nodes import PadAndTranslateImageForOutpainting
from .nodes.mask_nodes import PadForOutpaintGeneral

NODE_CONFIG = {
    "MultiCutAndDragWithTruck": {"class": MultiCutAndDragWithTruck, "name": "MultiCutAndDragWithTruck"},
    "BatchImageToMask": {"class": BatchImageToMask, "name": "BatchImageToMask"},
    "MapTrajectoriesToSegmentedMasks": {"class": MapTrajectoriesToSegmentedMasks, "name": "MapTrajectoriesToSegmentedMasks"},
    "PadAndTranslateImageForOutpainting": {"class": PadAndTranslateImageForOutpainting, "name": "PadAndTranslateImageForOutpainting"},
    "PadForOutpaintGeneral": {"class": PadForOutpaintGeneral, "name": "PadForOutpaintGeneral"}
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
