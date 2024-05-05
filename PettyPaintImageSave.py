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


class cstr(str):
    class color:
        END = "\33[0m"
        BOLD = "\33[1m"
        ITALIC = "\33[3m"
        UNDERLINE = "\33[4m"
        BLINK = "\33[5m"
        BLINK2 = "\33[6m"
        SELECTED = "\33[7m"

        BLACK = "\33[30m"
        RED = "\33[31m"
        GREEN = "\33[32m"
        YELLOW = "\33[33m"
        BLUE = "\33[34m"
        VIOLET = "\33[35m"
        BEIGE = "\33[36m"
        WHITE = "\33[37m"

        BLACKBG = "\33[40m"
        REDBG = "\33[41m"
        GREENBG = "\33[42m"
        YELLOWBG = "\33[43m"
        BLUEBG = "\33[44m"
        VIOLETBG = "\33[45m"
        BEIGEBG = "\33[46m"
        WHITEBG = "\33[47m"

        GREY = "\33[90m"
        LIGHTRED = "\33[91m"
        LIGHTGREEN = "\33[92m"
        LIGHTYELLOW = "\33[93m"
        LIGHTBLUE = "\33[94m"
        LIGHTVIOLET = "\33[95m"
        LIGHTBEIGE = "\33[96m"
        LIGHTWHITE = "\33[97m"

        GREYBG = "\33[100m"
        LIGHTREDBG = "\33[101m"
        LIGHTGREENBG = "\33[102m"
        LIGHTYELLOWBG = "\33[103m"
        LIGHTBLUEBG = "\33[104m"
        LIGHTVIOLETBG = "\33[105m"
        LIGHTBEIGEBG = "\33[106m"
        LIGHTWHITEBG = "\33[107m"

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(
                    f"'cstr' object already contains a code with the name '{name}'."
                )

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)


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


class PettyPaintImageSave:
    def __init__(self):
        self.type = "output"

    @classmethod
    def VALIDATE_INPUTS(s, images, filepath, overridefilepath, extension, quality, **_):
        print(images)
        print(filepath)
        print(overridefilepath)
        print(extension)
        print(quality)
        if filepath is None:
            return f"Invalid filepath: {filepath}"

        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filepath": ("STRING", {"default": "D:\\apainter\\layers"}),
                "overridefilepath": ("STRING", {"default": "", "forceInput": True}),
                "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"],),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "APorter"

    def save_images(
        self, images, filepath, overridefilepath, extension="png", quality=100
    ):
        if overridefilepath:
            file_path = os.path.join(overridefilepath,"images")
        else:
            file_path = filepath

        file_output_path = os.path.join(
            file_path, f"petty_paint_{generate_random_string()}.{extension}"
        )
        # Define token system
        print(file_path)
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
