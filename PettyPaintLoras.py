# Edited for Petty Paint.
# ---------------------------------------------------------------------------------------------------------------------#
# Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# for ComfyUI                                                 https://github.com/comfyanonymous/ComfyUI
# ---------------------------------------------------------------------------------------------------------------------#

import os
import sys
import comfy.sd
import comfy.utils
import folder_paths
import hashlib
from random import random, uniform
from .categories import icons

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def is_empty_or_whitespace(s):
    if s == None:
        return True
    return not s.strip()


# ---------------------------------------------------------------------------------------------------------------------#
# Based on Efficiency Nodes
# This is a lora stack where a single node has 3 different loras each with their own switch
class PettyPaintLoras:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "lora_setups":  ("STRING", {"default": "", "forceInput": True}),
                "switch_1": (["Off", "On"],),
                "lora_name_1": (loras,),
                "model_weight_1": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "clip_weight_1": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "switch_2": (["Off", "On"],),
                "lora_name_2": (loras,),
                "model_weight_2": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "clip_weight_2": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "switch_3": (["Off", "On"],),
                "lora_name_3": (loras,),
                "model_weight_3": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "clip_weight_3": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "lora_name_1_override": ("STRING", {"default": "", "forceInput": True}),
                "lora_name_2_override": ("STRING", {"default": "", "forceInput": True}),
                "lora_name_3_override": ("STRING", {"default": "", "forceInput": True}),
                "lora_stack": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = (
        "LORA_STACK",
        "STRING",
    )
    RETURN_NAMES = (
        "LORA_STACK",
        "show_help",
    )
    FUNCTION = "lora_stacker"
    CATEGORY = icons.get("Comfyroll/LoRA")

    def lora_stacker(
        self,
        lora_setups,
        lora_name_1,
        model_weight_1,
        clip_weight_1,
        switch_1,
        lora_name_2,
        model_weight_2,
        clip_weight_2,
        switch_2,
        lora_name_3,
        model_weight_3,
        clip_weight_3,
        switch_3,
        lora_stack=None,
        lora_name_1_override=None,
        lora_name_2_override=None,
        lora_name_3_override=None,
    ):
        # Initialise the list
        lora_list = list()
        if not is_empty_or_whitespace(lora_name_1_override):
            print("lora_name_1")
            print(lora_name_1)
            print("lora_name_1_override")
            print(lora_name_1_override)
            print(f"{len(lora_name_1_override)}")
            lora_name_1 = lora_name_1_override
            print(lora_name_1)
        if not is_empty_or_whitespace(lora_name_2_override):
            lora_name_2 = lora_name_2_override
        if not is_empty_or_whitespace(lora_name_3_override):
            lora_name_3 = lora_name_3_override

        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])

        if lora_name_1 != "None" and switch_1 == "On":
            lora_list.extend([(lora_name_1, model_weight_1, clip_weight_1)]),

        if lora_name_2 != "None" and switch_2 == "On":
            lora_list.extend([(lora_name_2, model_weight_2, clip_weight_2)]),

        if lora_name_3 != "None" and switch_3 == "On":
            lora_list.extend([(lora_name_3, model_weight_3, clip_weight_3)]),

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/LoRA-Nodes#cr-lora-stack"

        return (
            lora_list,
            show_help,
        )
