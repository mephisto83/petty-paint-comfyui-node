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

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, False, 0, 0, )

class PettyPaintKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "skip": ("BOOLEAN", {"default": False, "forceInput": True }),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff ,"forceInput": True}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000 ,"forceInput": True}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1 ,"forceInput": True, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS ,{"forceInput": True}, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS ,{"forceInput": True}, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT", "BOOLEAN", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "skip", "width", "height", )
    FUNCTION = "sample"

    CATEGORY = "PettyPaint"

    def sample(self, model, skip, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        if skip:
            latent_shape = latent_image["samples"].shape
            batch_size, channels, latent_height, latent_width = latent_shape
            height = latent_height * 8
            width = latent_width * 8
            return (latent_image, skip, width, height)     
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

class PettyPaintKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "skip": ("BOOLEAN", {"default": False, "forceInput": True }),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "forceInput": True}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "forceInput": True}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS ,{"forceInput": True}, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS ,{"forceInput": True}, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "forceInput": True}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT", "BOOLEAN", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "skip", "width", "height", )
    FUNCTION = "sample"

    CATEGORY = "PettyPaint"

    def sample(self, model, skip, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
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
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

class PettyPaintVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
            "skip": ("BOOLEAN", {"default": False, "forceInput": True }),
            "samples": ("LATENT", ), 
            "vae": ("VAE", )
            },
            "optional": {
                "width": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "height": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),                
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, skip, vae, samples, width=10, height=10):
        if skip:
            return (self.create_small_image( width, height), skip, )
        return (vae.decode(samples["samples"]), skip, )
    
    def create_small_image(self,  width, height):
        # Create a 10x10 image with a placeholder color (e.g., black)
        image_array = Image.new("RGB", (width, height), "black")
        image_array = np.array(image_array).astype(np.float32) / 255.0
        image_array = torch.from_numpy(image_array)[None,]
        return image_array