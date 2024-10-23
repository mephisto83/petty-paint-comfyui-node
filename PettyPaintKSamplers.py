import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging
from nodes import VAEEncode
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

from .PettyPaintText import any

class PETTYARGS(str):
    def __ne__(self, __value: object) -> bool:
        return False


pettyargs = PETTYARGS("*")


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (
        out,
        False,
        0,
        0,
    )

class SkippableVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "pixels": ("IMAGE", ), "vae": ("VAE", )},
            "optional": {"skip": (
                    any, 
                    {"forceInput": True}
                )}
        }
    # Optionally, you can override the RETURN_TYPES and FUNCTION if needed
    RETURN_TYPES = ("LATENT", "BOOLEAN",)
    FUNCTION = "encode"
    CATEGORY = "PettyPaint"

    def encode(self, vae, pixels, skip=False):
        if skip:
            return ({"samples": None}, )
        # Optionally, extend the functionality of the original encode method
        latent = VAEEncode().encode(vae, pixels)[0]["samples"]

        return ({"samples": latent}, skip, )

class ConvertWhiteToAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "pixels": ("IMAGE", ) },
            "optional": {
                "skip": (
                    any, 
                    {"forceInput": True}
                ),
                "color": (
                    any, 
                    {"forceInput": True}
                )
            }
        }
    # Optionally, you can override the RETURN_TYPES and FUNCTION if needed
    RETURN_TYPES = ("IMAGE", "BOOLEAN",)
    FUNCTION = "alpha_it"
    CATEGORY = "PettyPaint"

    def alpha_it(self, pixels, skip=False, color=None):
        if skip:
            return (pixels, skip, )
        # Ensure `pixels` is a list
        if not isinstance(pixels, (list, tuple)):
            pixels = [pixels]
        # Optionally, extend the functionality of the original encode method
        alphaed_ = []
        for pixel in pixels:
            alpha = convert_white_to_alpha(pixel, color)
            alphaed_.append(alpha)

        return (alphaed_, skip, )

    def convert_white_to_alpha(self, pixel):
        # Debug: Print the tensor shape before processing
        print(f"Processing tensor shape: {pixel.shape}")

        # Ensure the pixel is in a valid format (e.g., [B, H, W, C])
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[-1] == 3:  # Assuming shape [B, H, W, C]
                pixel = pixel.squeeze(0)  # Remove the batch dimension
                image_data = pixel.mul(255).byte().cpu().numpy()
            elif pixel.dim() == 3 and pixel.shape[-1] == 3:  # Assuming [H, W, C] format
                image_data = pixel.mul(255).byte().cpu().numpy()
            else:
                raise ValueError(f"Unsupported tensor dimension for image conversion: {pixel.shape}")

        # Ensure that the image data is 3D and in the correct shape for an image
        if len(image_data.shape) != 3 or image_data.shape[-1] != 3:
            raise ValueError(f"Unexpected shape after processing: {image_data.shape}")

        # Convert to PIL Image
        image = Image.fromarray(image_data)

        # Convert the image to RGBA if not already in that mode
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Efficiently convert white pixels to transparent using numpy and PIL
        image_np = np.array(image)
        # Create a mask where all white pixels are True
        white_mask = (image_np[:, :, :3] == [255, 255, 255]).all(axis=-1)
        # Set alpha channel to 0 for white pixels
        image_np[white_mask, 3] = 0

        # Convert back to an image
        image = Image.fromarray(image_np, "RGBA")

        # Convert back to tensor
        alphaed_pixel = torch.tensor(np.array(image).transpose(2, 0, 1)).float().div(255)

        return alphaed_pixel

def convert_white_to_alpha(image_tensor, color = None):
    """
    Convert white pixels in an image tensor to transparent.
    
    Parameters:
        image_tensor (torch.Tensor): Input tensor of shape [1, H, W, 3].
        
    Returns:
        torch.Tensor: Output tensor with shape [H, W, 4] where white pixels
                      are converted to transparent (0 alpha).
    """
    # Check that the input tensor has the expected shape
    if image_tensor.dim() != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[-1] != 3:
        raise ValueError("Expected input tensor shape [1, H, W, 3]")

    # Remove the batch dimension (squeeze)
    image_tensor = image_tensor.squeeze(0)  # Now shape is [H, W, 3]

    # Create a new tensor for the output with an additional alpha channel
    height, width = image_tensor.shape[:2]
    transparent_tensor = torch.zeros((height, width, 4), dtype=torch.uint8)

    # if color == None:
    #     color = {"r": 0, "b": 0, "g": 0}
    # Iterate over each pixel
    print(f"height: {height}, width: {width}")
    whiteness = 239
    for h in range(height):  # Iterate over height
        for w in range(width):  # Iterate over width
            r, g, b = image_tensor[h, w]  # Get the RGB values at (h, w)
            if r*255.0 > whiteness and g*255.0 > whiteness and b*255.0 > whiteness:
                # Set to transparent (0 alpha) if the pixel is white
                transparent_tensor[h, w, 0] = 255
                transparent_tensor[h, w, 1] = 255
                transparent_tensor[h, w, 2] = 255
                transparent_tensor[h, w, 3] = 0
            else:
                # Preserve the original color with full opacity (alpha = 255)
                if color != None:
                    transparent_tensor[h, w, 0] = color["r"]
                    transparent_tensor[h, w, 1] = color["g"]
                    transparent_tensor[h, w, 2] = color["b"]
                    transparent_tensor[h, w, 3] = 255
                else:
                    transparent_tensor[h, w, 0] = r
                    transparent_tensor[h, w, 1] = g
                    transparent_tensor[h, w, 2] = b
                    transparent_tensor[h, w, 3] = 255

    return transparent_tensor

class PettyPaintKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip": ("BOOLEAN", {"default": False, "forceInput": True}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "forceInput": True},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "forceInput": True,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"forceInput": True},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"forceInput": True},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "BOOLEAN",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "LATENT",
        "skip",
        "width",
        "height",
    )
    FUNCTION = "sample"

    CATEGORY = "PettyPaint"

    def sample(
        self,
        model,
        skip,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        if skip:
            latent_shape = latent_image["samples"].shape
            batch_size, channels, latent_height, latent_width = latent_shape
            height = latent_height * 8
            width = latent_width * 8
            return (latent_image, skip, width, height)
        return common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )


class PPKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "args": (pettyargs, {"default": None, "forceInput": True}),
            },
            "optional": {},
        }

    RETURN_TYPES = (
        "LATENT",
        "BOOLEAN",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "LATENT",
        "skip",
        "width",
        "height",
    )
    FUNCTION = "sample"
    CATEGORY = "PettyPaint"

    def sample(self, model, positive, negative, latent_image, args):
        print("---- PPK Sampler ----")
        cfg = args.get("cfg", 8.0)
        noise_seed = args.get("noise_seed", 0)
        sampler_name = args.get("sampler_name", "")
        scheduler = args.get("scheduler", "")
        start_at_step = args.get("start_at_step", 0)
        end_at_step = args.get("end_at_step", 10000)
        denoise = args.get("denoise", 0)
        add_noise = args.get("add_noise", "enable")
        return_with_leftover_noise = args.get("return_with_leftover_noise", "disable")
        steps = args.get("steps", 20)
        skip = args.get("skip", False)
        if skip:
            print("---- skipping ----")
            latent_shape = latent_image["samples"].shape
            batch_size, channels, latent_height, latent_width = latent_shape
            height = latent_height * 8
            width = latent_width * 8
            return (latent_image, skip, width, height)
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        print(f"---- {json.dumps(args)} ----")
        return common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )


class PettyPaintArguments:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "add_noise":  ("STRING", {"default": "enable", "forceInput": True}),
                "skip": (
                    any, 
                    {"forceInput": True}
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "forceInput": True},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "forceInput": True,
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"forceInput": True},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"forceInput": True},
                ),
                "start_at_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "forceInput": True},
                ),
                "end_at_step": (
                    "INT",
                    {"default": 10000, "min": 0, "max": 10000, "forceInput": True},
                ),
                "return_with_leftover_noise":  ("STRING", {"default": "enable", "forceInput": True}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (pettyargs,)
    RETURN_NAMES = ("PETTYARGS",)
    FUNCTION = "sample"

    CATEGORY = "PettyPaint"

    def sample(
        self,
        add_noise,
        skip,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise,
    ):
        return (
            {
                "skip": skip,
                "add_noise": add_noise,
                "noise_seed": noise_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_at_step": start_at_step,
                "end_at_step": end_at_step,
                "return_with_leftover_noise": return_with_leftover_noise,
                "denoise": denoise,
            },
        )


class PettyPaintKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip": ("BOOLEAN", {"default": False, "forceInput": True}),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "forceInput": True},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "forceInput": True,
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"forceInput": True},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"forceInput": True},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "forceInput": True},
                ),
                "end_at_step": (
                    "INT",
                    {"default": 10000, "min": 0, "max": 10000, "forceInput": True},
                ),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "BOOLEAN",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "LATENT",
        "skip",
        "width",
        "height",
    )
    FUNCTION = "sample"

    CATEGORY = "PettyPaint"

    def sample(
        self,
        model,
        skip,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        if skip:
            latent_shape = latent_image["samples"].shape
            batch_size, channels, latent_height, latent_width = latent_shape
            height = latent_height * 8
            width = latent_width * 8
            return (latent_image, skip, width, height)
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
        )


class PettyPaintVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "skip": ("BOOLEAN", {"default": False, "forceInput": True}),
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "IMAGE",
        "skip",
    )
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, skip, vae, samples, width=10, height=10):
        if skip:
            if width < 1:
                width = 10
            if height < 1:
                height = 10
            return (
                self.create_small_image(width, height),
                skip,
            )
        return (
            vae.decode(samples["samples"]),
            skip,
        )

    def create_small_image(self, width, height):
        # Create a 10x10 image with a placeholder color (e.g., black)
        image_array = Image.new("RGB", (width, height), "black")
        image_array = np.array(image_array).astype(np.float32) / 255.0
        image_array = torch.from_numpy(image_array)[None,]
        return image_array
