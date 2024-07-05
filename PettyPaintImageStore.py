# Image Save (NSP Compatible)
# Originally From ComfyUI/nodes.py
# SET TEXT TYPE
import os
from PIL import (
    Image,
    ImageFilter,
    ImageEnhance,
    ImageOps,
    ImageDraw,
    ImageChops,
    ImageFont,
)
from PIL.PngImagePlugin import PngInfo
import re
import numpy as np
import json
import random
import string


def generate_random_string(
    length=10, use_upper=True, use_lower=True, use_digits=True, use_special=False
):
    """Generate a random string of specified length and character types."""
    characters = ""
    if use_upper:
        characters += string.ascii_uppercase
    if use_lower:
        characters += string.ascii_lowercase
    if use_digits:
        characters += string.digits
    if use_special:
        characters += string.punctuation

    # Ensure character pool is not empty
    if not characters:
        raise ValueError("At least one character type must be selected")

    # Generate random string
    return "".join(random.choice(characters) for _ in range(length))


TEXT_TYPE = "STRING"


def check_file_exists(file_path):
    return os.path.exists(file_path)


def read_string_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None


ALLOWED_EXT = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")


class PettyPaintImageStore:
    def __init__(self):
        self.type = "output"


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filepath": ("STRING", {"default": ""}),
                "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "skip": ("BOOLEAN", {"default": False, "forceInput": True }),
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "PettyPaint"

    def save_images(
        self, images, filepath, extension="png", quality=100, skip=False
    ):
        if skip:
            return {"ui": {"images": []}}
        if filepath == "":
            return {"ui": {"images": []}}
        file_output_path = filepath
        # Define token system
        print(file_output_path)
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Delegate metadata/pnginfo
            if extension == "webp":
                pass
            else:
                metadata = PngInfo()
                exif_data = metadata

            # Save the images
            try:
                output_file = file_output_path
                if extension in ["jpg", "jpeg"]:
                    img.save(output_file, quality=quality, optimize=True)
                elif extension == "png":
                    img.save(output_file, pnginfo=exif_data, optimize=True)
                elif extension == "bmp":
                    img.save(output_file)
                elif extension == "tiff":
                    img.save(output_file, quality=quality, optimize=True)
                else:
                    img.save(output_file, pnginfo=exif_data, optimize=True)

            except OSError as e:
                print(e)
            except Exception as e:
                print(e)

        return {"ui": {"images": []}}
