# Edited for Petty Paint.
# ---------------------------------------------------------------------------------------------------------------------#
# Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# for ComfyUI                                                 https://github.com/comfyanonymous/ComfyUI
# ---------------------------------------------------------------------------------------------------------------------#

import os
import nodes
import hashlib
import sys
import torch
import numpy as np

from PIL import Image, ImageOps, ImageSequence
from torch import negative_
from nodes import MAX_RESOLUTION
import torch.nn.functional as F

import folder_paths
import node_helpers
import comfy.utils

import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")

def is_empty_or_whitespace(s):
    if s == None:
        return True
    return not s.strip()


# ---------------------------------------------------------------------------------------------------------------------#
# Based on Efficiency Nodes
# This is a lora stack where a single node has 3 different loras each with their own switch
class PettyPaintText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "STRING",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(
        self,
        text
    ):
        return (
            text,
        )

# ---------------------------------------------------------------------------------------------------------------------#
# Based on Efficiency Nodes
# This is a lora stack where a single node has 3 different loras each with their own switch
class PettyPaintRemoveAddText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                "remove":  ("STRING", {"default": "", "multiline": True}),
                "add":  ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "STRING",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text, remove, add):
        # Split the remove and add strings into lists
        remove_list = remove.split(';')
        add_list = add.split(';')

        # Remove each substring in remove_list from text
        for item in remove_list:
            text = text.replace(item, '')

        # Add each substring in add_list to the end of text with a preceding space and comma
        for item in add_list:
            if item:  # This checks if the item is not empty
                text += ', ' + item

        return (text, )


# ---------------------------------------------------------------------------------------------------------------------#
# Based on Efficiency Nodes
# This is a lora stack where a single node has 3 different loras each with their own switch
class PettyPaintAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                "append":  ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "STRING",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text, append):
        # Split the remove and add strings into lists

        # Remove each substring in remove_list from tex
        text = f"{text}{append}"
        return (text, )

class PettyPaintJsonRead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  (any, {}),
                "path":  (any, {}),
            },
        }
    RETURN_TYPES = (
        any,
    )
    RETURN_NAMES = (
        "data",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    @classmethod
    def IS_CHANGED(self, text, path):
        # Split the remove and add strings into lists

        # Remove each substring in remove_list from tex
        if isinstance(text, str):
            data = json.loads(text)  # This will throw an error if text is not valid JSON
        else:
            data = text
        
        # Split the path string into keys
        keys = path.split('.')
        
        # Traverse the JSON object according to the path
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]  # Move to the next level in the JSON structure
        return (json.dumps(data))
    
    def doStuff(self, text, path):
        # Split the remove and add strings into lists

        # Remove each substring in remove_list from tex
        if isinstance(text, str):
            data = json.loads(text)  # This will throw an error if text is not valid JSON
        else:
            data = text
        
        # Split the path string into keys
        keys = path.split('.')
        
        # Traverse the JSON object according to the path
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]  # Move to the next level in the JSON structure
        print(json.dumps(data))
        if isinstance(data, dict):
            return {"ui": {"text": text}, "result": (data, )}
        return {"ui": {"text": text}, "result": (data, )}

class PettyPaintJsonReadArray:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                "path":  ("INT", {"default": 0,  "min": 0, "max": 100 }),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "STRING",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text, path):
        if isinstance(text, str):
            data = json.loads(text)  # This will throw an error if text is not valid JSON
        else:
            data = text
        print(json.dumps(data[int(path)]))
        return {"ui": {"text": text}, "result": (data[path], )}
    
class PettyPaintTexts_to_Conditioning:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "texts": (any, {}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, clip, texts):
        result = []
        for text in texts:
            encoder = nodes.CLIPTextEncode()
            encoded = encoder.encode(clip=clip, text=text)
            result.append(encoded[0])
        return (result, )

class PettyPaintLoadImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": (any, {}),
                }
        }
        
    @classmethod
    def IS_CHANGED(s, images):
        separator = " "
        separator.join(images)


    CATEGORY = "PettyPaint"
    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, images):
        result_images = []
        result_masks = []
        for image_path in images:
            if os.path.exists(image_path):
                img = node_helpers.open_image(image_path)
                
                output_images = []
                output_masks = []
                for i in ImageSequence.Iterator(img):
                    i = ImageOps.exif_transpose(i)
                    if i.mode == 'I':
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    if 'A' in i.getbands():
                        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                    output_images.append(image)
                    output_masks.append(mask.unsqueeze(0))

                if len(output_images) > 1:
                    output_image = torch.cat(output_images, dim=0)
                    output_mask = torch.cat(output_masks, dim=0)
                else:
                    output_image = output_images[0]
                    output_mask = output_masks[0]
                result_images.append(output_image)
                result_masks.append(output_mask)

        return (result_images, result_masks, )

class PettyPaintLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{ 
                "imagepath":  (any, {}),    
            }}

    CATEGORY = "PettyPaint"
    RETURN_TYPES = ("IMAGE","IMAGE", "INT", "INT", )
    RETURN_NAMES = ("IMAGE","IMAGE", "WIDTH", "HEIGHT", )
    FUNCTION = "load_image"

    def load_image(self, imagepath):
        if os.path.exists(imagepath):
            img = node_helpers.open_image(imagepath)
            output_images = []
            output_masks = []
            width, height = img.size  # Get the dimensions of the image
            for i in ImageSequence.Iterator(img):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            return (output_image, output_mask, width, height,)
        return (None, None, 0 , 0)

class PettyPaintImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE", ),
                    "channel": (["red", "green", "blue", "alpha"],),
                }
        }

    CATEGORY = "PettyPaint"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"

    def image_to_mask(self, images, channel):
        result = []
        print(images)
        print(len(images))
        for image in images:
            channels = ["red", "green", "blue", "alpha"]
            mask = image[:, :, :, channels.index(channel)]
            result.append(mask)
        return (result, )

class PettyPaintJsonMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                "path":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "data",
        "STRING",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text, path):
        if isinstance(text, str):
            datas = json.loads(text)  # This will throw an error if text is not valid JSON
        else:
            datas = text
        # Split the path string into keys
        keys = path.split('.')
        
        # Traverse the JSON object according to the path
        result = []
        for data in datas:
            item = data
            for key in keys:
                if isinstance(item, dict) and key in item:
                    item = item[key]  # Move to the next level in the JSON structure
            result.append(item)
        return  (result, )
class PettyPaintEnsureDirectory:
    RETURN_TYPES = (any,  )
    RETURN_NAMES = ("data",)

    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":  (any, {}),
            },
        }
    def doStuff(self, value):
        self.ensure_directory_exists(value)
        return (value, )

    def ensure_directory_exists(self, path):
        """
        Ensures that the directory specified by 'path' exists. If the directory does not exist, it creates it.
        
        :param path: The filesystem path to the directory to check/create.
        """
        try:
            # Try to create the directory (it will do nothing if the directory already exists)
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' is ready.")
        except Exception as e:
            print(f"Failed to create or access directory '{path}'. Reason: {e}")
        return True

class PettyPaintProcessor:
    RETURN_TYPES = (any, "STRING", "STRING", any, "INT", comfy.samplers.KSampler.SAMPLERS, "FLOAT", any, "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("any", "output", "model", "vae", "steps", "samplers", "cfg", "scheduler", "prompt", "negative_prompt", "page_cell_path", "positivePrompts", "negativePrompts", )

    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":  (any, {}),
                "file_path":  ("STRING", { "forceInput": True, "default": "" }),
                "expected_files":  ("INT", {  "default": 4, "min": 0, "max": 120, "step": 1 }),
            },
        }

    def IS_CHANGED(self, value, file_path, expected_files):
        return True

    def doStuff(self, value, file_path, expected_files):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        temp = read_string_from_file(file_path)
        data = json.loads(temp)
        storage_path = data["storage"]
        character_data = data["data"]
        if "pages" in data:
            page_data = data["pages"] 
        else:
            page_data = []
        context = data["context"]
        context["page_prompt"] = context.get("page_prompt", "")
        context["page_negative_prompt"] = context.get("page_negative_prompt", "")
        context["page_model"] = context.get("page_model", "")
        context["model"] = context.get("model", "")

        
        changed = False
        output = ""
        try:
            for chpt in checkpoints:
                character_name = None
                for c_data in character_data:
                    expected_pose_files = -1
                    character_name = c_data["name"]
                    character_path = os.path.join(storage_path, chpt, character_name)
                    if "poses" in c_data:
                        expected_pose_files = len(c_data["poses"]) * 2
                    if os.path.exists(character_path):
                        total_files = count_files_recursively(character_path)
                        print(f"character_path {character_path}")
                        print(f"total_files {total_files}")
                        print(f"expected_pose_files {expected_pose_files}")
                        output += f"character_path {character_path}\n"
                        output += f"total_files {total_files}\n"
                        output += f"expected_pose_files {expected_pose_files}\n"
                        output += f"------------------------------------------\n"
                        if (total_files < expected_files and expected_pose_files == -1) or expected_pose_files > total_files:
                            context["current_character"] = character_data.index(c_data)
                            context["model"] = chpt
                            changed = True
                            break
                    else:
                        context["current_character"] = character_data.index(c_data)
                        context["model"] = chpt
                        changed = True
                        break
                page_number = None
                set_page = False
                for p_data in page_data:
                    cell_data = p_data["cells"]
                    for c_data in cell_data:
                        if not set_page:
                            page_number = "page_" + str(p_data["page_number"]) + "_" + str(c_data["cell_number"]) + ".png"
                            page_cell_path = os.path.join(storage_path, chpt, page_number)
                            if not os.path.exists(page_cell_path):
                                output += f"page_number {page_number}\n"
                                context["page_prompt"] = c_data["environment_prompt"]
                                context["page_negative_prompt"] = c_data["negative_prompt"]
                                context["page_model"] = chpt
                                context["model"] = chpt
                                set_page = True
                                changed = True
                                break
                if changed:
                    break
        except ValueError:
            print('Item not found in the list.')

        if not "page_defaults" in data:
            data["page_defaults"] = {}
            changed = True
        positivePrompts = ""
        negativePrompts = ""
        if "model_information" in data:
            model_information = data["model_information"]
            if context["model"] in model_information:
                info = model_information[context["model"]]
                if "positivePrompts" in info:
                    positivePrompts = " ".join(info["positivePrompts"])
                if "negativePrompts" in info:
                    negativePrompts = " ".join(info["negativePrompts"])

        if changed:
            if "model_information" in data:
                model_information = data["model_information"]
                if context["model"] in model_information:
                    info = model_information[context["model"]]
                    if "technicalDetails" in info:
                        technicalDetails = info["technicalDetails"]
                        if "steps" in technicalDetails:
                           data["character_defaults"]["steps"] = int(technicalDetails["steps"])
                        if "sampler" in technicalDetails:
                           data["character_defaults"]["sampler_name"] = technicalDetails["sampler"]
                        if "steps" in technicalDetails:
                           data["character_defaults"]["cfg"] = float(technicalDetails["cfgScale"])
                        if "scheduler" in technicalDetails:
                           data["character_defaults"]["scheduler"] = technicalDetails["scheduler"]
            if file_path:
                write_string_to_file(json.dumps(data, indent=4, sort_keys=True), file_path)
        res = json.dumps(data)
        return (
            res, 
            output,
            context["model"],
            data.get("model_information", {}).get(data.get("context", {}).get("model", {}), {}).get("vae", "sdxlVAE_sdxlVAE.safetensors"),
            data["character_defaults"]["steps"],  
            data["character_defaults"]["sampler_name"], 
            data["character_defaults"]["cfg"], 
            data["character_defaults"]["scheduler"],
            context["page_prompt"] + " " + positivePrompts,
            context["page_negative_prompt"] + " " + negativePrompts,
            page_cell_path,
            positivePrompts,
            negativePrompts
        )


class PettyPaintCountFiles:
    RETURN_TYPES = (any,  )
    RETURN_NAMES = ("INT",)

    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":  (any, {}),
            },
        }

    def doStuff(self, value):
        count = self.count_files_recursively(value)
        return (count, )

    def count_files_recursively(self, folder_path):
        """
        Recursively counts the number of files in the specified folder, including all subdirectories.

        :param folder_path: The path to the folder in which to count files.
        :return: The total number of files in the folder and its subdirectories.
        """
        file_count = 0
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(folder_path):
            file_count += len(files)

        return file_count
class PettyPaintExec:
    @classmethod
    def IS_CHANGED(self, value, code, option = None):
        datas = value
        
        # Traverse the JSON object according to the path
        print("-------------- Petty Paint exec")
        print(code)
        print(isinstance(datas, str))
        
        res = eval(code, {'item': datas, "option": option})
        
        print(res)
        print("-------------- Petty Paint exec end")
        return (res, ) 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":  (any, {}),
                "code":  ("STRING", {"default": "" ,"multiline": True}),
                
            },
            "optional": {
                "option":  (any, {}),
            }
        }

    RETURN_TYPES = (any,  )
    RETURN_NAMES = ("data",)

    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, value, code, option = None):
        datas = value
        
        # Traverse the JSON object according to the path
        print("-------------- Petty Paint exec")
        print(code)
        print(datas)
        print(isinstance(datas, str))
        
        res = eval(code, {
            'item': datas, 
            "option": option,
            'os': os
            })
        
        print(res)
        print("-------------- Petty Paint exec end")
        return (res, ) 

class PettyPaintToJson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
            }
        }

    RETURN_TYPES = (
        any,
    )
    RETURN_NAMES = (
        "data",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text):
        if isinstance(text, str):
            data = json.loads(text)  # This will throw an error if text is not valid JSON
            print("converted string to json")
        else:
            data = text
        print(data)
        return (data, )   

class PettyPaintConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":  (any, {}),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "data",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, value):
        data = json.dumps(value)
        return (data, )
    
class PettyPaintMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":  (any, {}),
                "code":  ("STRING", {"default": "" ,"forceInput": True, "multiline": False}),
                
            },
            "optional": {
                "option":  (any, {}),
            }
        }

    RETURN_TYPES = (
        any,
    )
    RETURN_NAMES = (
        "data",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, text, code, option = None):
        if isinstance(text, str):
            datas = json.loads(text)  # This will throw an error if text is not valid JSON
        else:
            datas = text
        
        # Traverse the JSON object according to the path
        result = []
        print("-------------- Petty Paint map")
        for index in range(len(datas)):
            item = datas[index]
            res = eval(code, {'item': item, "option": option, "index": index, 'list': datas})
            
            print(res)
            result.append(res)
        print("-------------- Petty Paint map end")
        return (result, )  

class PettyPaintConditioningSetMaskAndCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positives": ("CONDITIONING", ),
                "negatives": ("CONDITIONING", ),
                "masks": ("MASK", ),
                # "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                # "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"
    DESCRIPTION = """
Bundles multiple conditioning mask and combine nodes into one,functionality is identical to ComfyUI native nodes
"""

    def append(self, positives, negatives, masks, set_cond_area):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True

        for index in range(len(positives)):
            mask_1 = masks[index]
            if len(mask_1.shape) < 3:
                mask_1 = mask_1.unsqueeze(0)
            mask_1_strength = 1
            positive_1 = positives[index]
            for t in positive_1:
                append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)

        for index in range(len(negatives)):
            mask_1 = masks[index]
            if len(mask_1.shape) < 3:
                mask_1 = mask_1.unsqueeze(0)
            mask_1_strength = 1
            negative_1 = negatives[index]
            for t in negative_1:
                append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        return (c, c2)

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    print(destination.shape)
    print(source.shape)
    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

def combine_mask(destination, source, x, y, operation = "add"):
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1]))

    left, top = (x, y,)
    right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
    visible_width, visible_height = (right - left, bottom - top,)
    print("top")
    print(top)
    print("bottom")
    print(bottom)
    print("left")
    print(left)
    print("right")
    print(right)
    print("visible_height")
    print(visible_height)
    print("visible_width")
    print(visible_width)
    print(source)
    print(destination)
    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = destination[:, top:bottom, left:right]

    if operation == "multiply":
        output[:, top:bottom, left:right] = destination_portion * source_portion
    elif operation == "add":
        output[:, top:bottom, left:right] = destination_portion + source_portion
    elif operation == "subtract":
        output[:, top:bottom, left:right] = destination_portion - source_portion
    elif operation == "and":
        output[:, top:bottom, left:right] = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
    elif operation == "or":
        output[:, top:bottom, left:right] = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
    elif operation == "xor":
        output[:, top:bottom, left:right] = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()

    output = torch.clamp(output, 0.0, 1.0)

    return (output,)

class PettyPaintImageCompositeMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "sources": ("IMAGE",),
                "masks": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, destination, sources, masks, x, y, resize_source):
        print(destination)
        print(sources)
        print(masks)
        print(len(destination))
        print(len(sources))
        print(len(masks))
        raw_masks = [mask for mask in masks]

        for index in range(len(sources)):
            destination = destination.clone().movedim(-1, 1)
            source = sources[index]
            # if index > 0:
            #     comp_mask = combine_mask(comp_mask, raw_masks[index], x, y) 
            # else:
            comp_mask = raw_masks[index][0]
            destination = composite(destination, source.movedim(-1, 1), x, y, comp_mask, 1, resize_source).movedim(1, -1)
        return (destination,)


def append_helper(t, mask, c, set_area_to_bounds, strength):
        n = [t[0], t[1].copy()]
        _, h, w = mask.shape
        n[1]['mask'] = mask
        n[1]['set_area_to_bounds'] = set_area_to_bounds
        n[1]['mask_strength'] = strength
        c.append(n)  


class PettyPaintImagePlacement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_destination": ("IMAGE",),
                "canvas_width": ("INT", { "forceInput": True, }),
                "canvas_height": ("INT", { "forceInput": True }),
                "destination":  ("STRING", {"default": "top-center" ,"forceInput": True, "multiline": False}),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "PettyPaint"

    def execute(self, image, image_destination, canvas_width, canvas_height, destination, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        print("canvas_height")
        print(canvas_height)
        print("canvas_width")
        print(canvas_width)
        print("image.shape[3]")
        print(image.shape[3])
        canvas = image_destination.clone()

        dest_x, dest_y = self._calculate_position(canvas_width, canvas_height, ow, oh, destination, x_offset, y_offset)

        # Calculate the bounds for the placement to ensure the image fits within the canvas
        top = max(0, dest_y)
        left = max(0, dest_x)
        bottom = min(dest_y + oh, canvas_height)
        right = min(dest_x + ow, canvas_width)

        # Placing the image on the canvas
        # canvas[top:bottom, left:right, :] = image[:, max(0, -dest_y):oh + min(0, canvas_height - dest_y), max(0, -dest_x):ow + min(0, canvas_width - dest_x), :]

        res = composite(canvas.movedim(-1, 1), image.movedim(-1, 1), left, top, None, 1, False).movedim(1, -1)
        return res,

    def _calculate_position(self, cw, ch, iw, ih, position, x_offset, y_offset):
        if "center" in position:
            x = (cw - iw) // 2
            y = (ch - ih) // 2
        elif "top" in position:
            y = 0
            x = (cw - iw) // 2 if "center" in position else 0 if "left" in position else cw - iw
        elif "bottom" in position:
            y = ch - ih
            x = (cw - iw) // 2 if "center" in position else 0 if "left" in position else cw - iw
        elif "left" in position:
            x = 0
            y = (ch - ih) // 2
        elif "right" in position:
            x = cw - iw
            y = (ch - ih) // 2

        x += x_offset
        y += y_offset

        # Adjusting the coordinates to ensure they remain within the image dimensions
        x = max(0, min(x, cw - iw))
        y = max(0, min(y, ch - ih))

        return x, y

    def generate_empty_image(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )
    

def count_files_recursively(folder_path):
    """
    Recursively counts the number of files in the specified folder, including all subdirectories.

    :param folder_path: The path to the folder in which to count files.
    :return: The total number of files in the folder and its subdirectories.
    """
    file_count = 0
    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)

    return file_count

def write_string_to_file(string, file_path):
    """
    Writes a given string to a file. If the file does not exist, it will be created.
    If the file exists, it will be overwritten.

    :param string: The string to write to the file.
    :param file_path: The path to the file where the string should be written.
    """
    try:
        # Open the file in write mode ('w'). This also creates the file if it doesn't exist.
        with open(file_path, 'w') as file:
            file.write(string)
        print(f"Successfully wrote to {file_path}.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    

class PettyPaintLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "setup": (any, ),
            },
             "optional": {
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
    CATEGORY = "PettyPaint"

    def lora_stacker(
        self,
        setup,
        lora_stack=None,
    ):
        # Initialise the list
        lora_list = list()
        existing_loras = folder_paths.get_filename_list("loras")

        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])
        if setup != None:
            for lora_setup in setup:
                name = lora_setup.get("name")
                model_weight = lora_setup.get("model_weight")
                clip_weight = lora_setup.get("clip_weight")
                if name in existing_loras and model_weight and clip_weight:
                    lora_list.extend([(name, float(model_weight), float(clip_weight))])
                    print("added lora")
                else:
                    print("skipped adding lora")

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/LoRA-Nodes#cr-lora-stack"

        return (
            lora_list,
            show_help,
        )


class PettyPaintApplyLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "STRING",
    )
    RETURN_NAMES = (
        "MODEL",
        "CLIP",
        "show_help",
    )
    FUNCTION = "apply_lora_stack"
    CATEGORY = "PettyPaint"

    def apply_lora_stack(
        self,
        model,
        clip,
        lora_stack=None,
    ):
        show_help = ""

        # Initialise the list
        lora_params = list()

        # Extend lora_params with lora-stack items
        if lora_stack:
            lora_params.extend(lora_stack)
        else:
            return (
                model,
                clip,
                show_help,
            )

        # Initialise the model and clip
        model_lora = model
        clip_lora = clip
        
        # Loop through the list
        for tup in lora_params:
            lora_name, strength_model, strength_clip = tup

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            show_help += lora_name + " model: " + str(strength_model) + "  clip:" + str(strength_clip)+ "\n"
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model_lora, clip_lora, lora, strength_model, strength_clip
            )

        return (
            model_lora,
            clip_lora,
            show_help,
        )



class PettyPaintFakeConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any,),
            }
        }

    RETURN_TYPES = (
        "INT",
        "STRING",
        "FLOAT",
        "COMBO",
        comfy.samplers.KSampler.SAMPLERS,
        any,
    )
    RETURN_NAMES = (
         "INT",
        "STRING",
        "FLOAT",
        "COMBO",
        "SAMPLERS",
        "any"
    )
    FUNCTION = "apply_lora_stack"
    CATEGORY = "PettyPaint"

    def apply_lora_stack(
        self,
        value,
    ):
        
        return (
            value,
            value,
            value,
            value,
            value,
            value,
        )
    
class PettyImageImageColorToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#ffffff", "display": "color"}),
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"

    def hex_to_int(self, hex_str):
        # Remove the '#' character if it exists
        hex_str = hex_str.lstrip('#')
        # Convert the hex string to an integer
        return int(hex_str, 16)

    def image_to_mask(self, image, color):
        # Convert color from hex string to integer
        color_int = self.hex_to_int(color)

        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]
        mask = torch.where(temp == color_int, 255, 0).float()
        return (mask,)

    
class PettyPaintImageColorsToMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("STRING",  {"default": "#ffffff", "forceInput": True, "display": "color"}),
            }
        }

    CATEGORY = "PettyPaint"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"

    def hex_to_int(self, hex_str):
        # Remove the '#' character if it exists
        hex_str = hex_str.lstrip('#')
        # Convert the hex string to an integer
        return int(hex_str, 16)

    def image_to_mask(self, image, colors):
        # Convert each color from hex string to integer
        color_ints = [self.hex_to_int(color) for color in colors]

        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]

        masks = []
        for color_int in color_ints:
            mask = torch.where(temp == color_int, 255, 0).float()
            masks.append(mask)

        return (masks, )

class PettyPaintMasksToImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "masks": ("MASK",),
                }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, masks):
        temp = []
        for mask in masks:
            result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            temp.append(result)
        return (temp,)

class PettyPaintImagesToMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "channel": (["red", "green", "blue", "alpha"],),
                }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"

    def image_to_mask(self, images, channel):
        results = []
        for image in images:
            channels = ["red", "green", "blue", "alpha"]
            mask = image[:, :, :, channels.index(channel)]
            results.append(mask)
        return (results,)

class PettyPaintBlurs:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "image/postprocessing"

    def blur(self, images: AnyType, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (images,)
        results = []
        for image in images:
            image = image.to(comfy.model_management.get_torch_device())
            batch_size, height, width, channels = image.shape

            kernel_size = blur_radius * 2 + 1
            kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

            image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
            padded_image = F.pad(image, (blur_radius,blur_radius,blur_radius,blur_radius), 'reflect')
            blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:,:,blur_radius:-blur_radius, blur_radius:-blur_radius]
            blurred = blurred.permute(0, 2, 3, 1)

            results.append(blurred.to(comfy.model_management.intermediate_device()))
        return (results, )

def gaussian_kernel(kernel_size: int, sigma: float, device=None):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device), torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()