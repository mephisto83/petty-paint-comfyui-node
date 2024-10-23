import os
import nodes
import hashlib
import sys
import torch
import numpy as np
import cv2
from nodes import MAX_RESOLUTION, SaveImage
from comfy_extras.nodes_mask import ImageCompositeMasked
from PIL import Image, ImageOps, ImageSequence
from torch import negative_
import torch.nn.functional as F

import folder_paths
import node_helpers
import comfy.utils

import json

ControlNetColors = {
    "joints": [
        {
            "id": 0,
            "name": "nose",
            "color": {"R": 207, "G": 1, "B": 1, "HEX": "#FF0000"},
        },
        {
            "id": 1,
            "name": "neck",
            "color": {"R": 255, "G": 85, "B": 0, "HEX": "#FF5500"},
        },
        {
            "id": 2,
            "name": "Right Shoulder",
            "color": {"R": 255, "G": 170, "B": 0, "HEX": "#FFAA00"},
        },
        {
            "id": 3,
            "name": "Right Elbow",
            "color": {"R": 255, "G": 255, "B": 0, "HEX": "#FFFF00"},
        },
        {
            "id": 4,
            "name": "Right Wrist",
            "color": {"R": 170, "G": 255, "B": 0, "HEX": "#AAFF00"},
        },
        {
            "id": 5,
            "name": "Left Shoulder",
            "color": {"R": 85, "G": 255, "B": 0, "HEX": "#55FF00"},
        },
        {
            "id": 6,
            "name": "Left Elbow",
            "color": {"R": 0, "G": 255, "B": 0, "HEX": "#00FF00"},
        },
        {
            "id": 7,
            "name": "Left Wrist",
            "color": {"R": 0, "G": 255, "B": 85, "HEX": "#00FF55"},
        },
        {
            "id": 8,
            "name": "Right Hip",
            "color": {"R": 0, "G": 255, "B": 170, "HEX": "#00FFAA"},
        },
        {
            "id": 9,
            "name": "Right Knee",
            "color": {"R": 0, "G": 255, "B": 255, "HEX": "#00FFFF"},
        },
        {
            "id": 10,
            "name": "Right Ankle",
            "color": {"R": 0, "G": 170, "B": 255, "HEX": "#00AAFF"},
        },
        {
            "id": 11,
            "name": "Left Hip",
            "color": {"R": 0, "G": 85, "B": 255, "HEX": "#0055FF"},
        },
        {
            "id": 12,
            "name": "Left Knee",
            "color": {"R": 0, "G": 0, "B": 255, "HEX": "#0000FF"},
        },
        {
            "id": 13,
            "name": "Left Ankle",
            "color": {"R": 85, "G": 0, "B": 255, "HEX": "#5500FF"},
        },
        {
            "id": 14,
            "name": "Right Eye",
            "color": {"R": 170, "G": 0, "B": 255, "HEX": "#AA00FF"},
        },
        {
            "id": 15,
            "name": "Left Eye",
            "color": {"R": 255, "G": 0, "B": 255, "HEX": "#FF00FF"},
        },
        {
            "id": 16,
            "name": "Right Ear",
            "color": {"R": 255, "G": 0, "B": 170, "HEX": "#FF00AA"},
        },
        {
            "id": 17,
            "name": "Left Ear",
            "color": {"R": 255, "G": 0, "B": 85, "HEX": "#FF0055"},
        },
    ],
    "bones": [
        {
            "pair": [1, 2],
            "name": "Right Shoulderblade",
            "color": {"R": 153, "G": 0, "B": 0, "HEX": "#990000", "TO": "#960000"},
        },
        {
            "pair": [1, 5],
            "name": "Left Shoulderblade",
            "color": {"R": 153, "G": 51, "B": 0, "HEX": "#a13b00", "TO": "#9a3600"},
        },
        {
            "pair": [2, 3],
            "name": "Right Arm",
            "color": {"R": 153, "G": 102, "B": 0, "HEX": "#885f00", "TO": "#9b7201"},
        },
        {
            "pair": [3, 4],
            "name": "Right Forearm",
            "color": {"R": 153, "G": 153, "B": 0, "HEX": "#999900"},
        },
        {
            "pair": [5, 6],
            "name": "Left Arm",
            "color": {"R": 102, "G": 153, "B": 0, "HEX": "#669900"},
        },
        {
            "pair": [6, 7],
            "name": "Left Forearm",
            "color": {"R": 51, "G": 153, "B": 0, "HEX": "#339900"},
        },
        {
            "pair": [1, 8],
            "name": "Right Torso",
            "color": {"R": 0, "G": 153, "B": 0, "HEX": "#009900"},
        },
        {
            "pair": [8, 9],
            "name": "Right Upper Leg",
            "color": {"R": 0, "G": 153, "B": 51, "HEX": "#009933"},
        },
        {
            "pair": [9, 10],
            "name": "Right Lower Leg",
            "color": {"R": 0, "G": 153, "B": 102, "HEX": "#009966"},
        },
        {
            "pair": [1, 11],
            "name": "Left Torso",
            "color": {"R": 0, "G": 153, "B": 153, "HEX": "#009999"},
        },
        {
            "pair": [11, 12],
            "name": "Left Upper Leg",
            "color": {"R": 0, "G": 102, "B": 153, "HEX": "#006699"},
        },
        {
            "pair": [12, 13],
            "name": "Left Lower Leg",
            "color": {"R": 0, "G": 51, "B": 153, "HEX": "#003399"},
        },
        {
            "pair": [1, 0],
            "name": "Head",
            "color": {"R": 0, "G": 0, "B": 153, "HEX": "#000099"},
        },
        {
            "pair": [0, 14],
            "name": "Right Eyebrow",
            "color": {"R": 51, "G": 0, "B": 153, "HEX": "#330099"},
        },
        {
            "pair": [14, 16],
            "name": "Right Ear",
            "color": {"R": 102, "G": 0, "B": 153, "HEX": "#660099"},
        },
        {
            "pair": [0, 15],
            "name": "Left Eyebrow",
            "color": {"R": 153, "G": 0, "B": 153, "HEX": "#990099"},
        },
        {
            "pair": [15, 17],
            "name": "Left Ear",
            "color": {"R": 153, "G": 0, "B": 102, "HEX": "#990066"},
        },
    ],
}


def get_legs_part_colors(control_net_colors):
    head_joints = [
        "Right Hip",
        "Right Knee",
        "Right Knee",
        "Right Ankle",
        "Left Knee",
        "Left Ankle",
    ]
    head_bones = [
        "Right Upper Leg",
        "Right Lower Leg",
        "Left Upper Leg",
        "Left Lower Leg",
    ]

    head_colors = []

    # Get colors for head joints
    for joint in control_net_colors["joints"]:
        if joint["name"] in head_joints:
            head_colors.append(joint["color"])

    # Get colors for head bones
    for bone in control_net_colors["bones"]:
        if bone["name"] in head_bones:
            head_colors.append(bone["color"])

    return head_colors


def get_head_part_colors(control_net_colors):
    head_joints = ["nose"]
    head_bones = ["nose"]

    head_colors = []

    # Get colors for head joints
    for joint in control_net_colors["joints"]:
        if joint["name"] in head_joints:
            head_colors.append(joint["color"])

    # Get colors for head bones
    for bone in control_net_colors["bones"]:
        if bone["name"] in head_bones:
            head_colors.append(bone["color"])

    return head_colors


def get_torso_part_colors(control_net_colors):
    head_joints = [
        "neck",
        "Right Shoulder",
        "Right Elbow",
        "Right Wrist",
        "Left Shoulder",
        "Left Elbow",
        "Left Wrist",
    ]
    head_bones = [
        "Right Shoulderblade",
        "Left Shoulderblade",
        "Right Arm",
        "Right Forearm",
        "Left Arm",
        "Left Forearm",
        "Right Torso",
        "Left Torso",
    ]

    head_colors = []

    # Get colors for head joints
    for joint in control_net_colors["joints"]:
        if joint["name"] in head_joints:
            head_colors.append(joint["color"])

    # Get colors for head bones
    for bone in control_net_colors["bones"]:
        if bone["name"] in head_bones:
            head_colors.append(bone["color"])

    return head_colors


# ---------------------------------------------------------------------------------------------------------------------#
# Based on Efficiency Nodes
# This is a lora stack where a single node has 3 different loras each with their own switch
class PettyPaintControlNetToMasking:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "STRING",
                    {"default": "", "forceInput": True, "multiline": False},
                ),
                "expansion_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "forceInput": True},
                ),
                "tolerance": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "head",
        "torso",
        "leg",
    )
    FUNCTION = "doStuff"
    CATEGORY = "PettyPaint"

    def doStuff(self, image, expansion_size, tolerance):
        # Split the remove and add strings into lists
        head_colors = get_head_part_colors(ControlNetColors)
        torso_colors = get_torso_part_colors(ControlNetColors)
        leg_colors = get_legs_part_colors(ControlNetColors)

        # Remove each substring in remove_list from tex
        head_image = expand_colors_with_uniform_tolerance(
            image, head_colors, tolerance, expansion_size
        )
        head_image = reload_image(head_image)
        torso_image = expand_colors_with_uniform_tolerance(
            image, torso_colors, tolerance, expansion_size
        )
        torso_image = reload_image(torso_image)
        leg_image = expand_colors_with_uniform_tolerance(
            image, leg_colors, tolerance, expansion_size
        )
        leg_image = reload_image(leg_image)

        return (head_image, torso_image, leg_image)


def reload_image(head_image):
    head_image.save("d:\\temp.png")
    return load_image_temp_image()


def load_image_temp_image(imagepath="d:\\temp.png"):
    if imagepath != None and os.path.exists(imagepath):
        img = Image.open(imagepath)  # node_helpers.open_image(imagepath)
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            return image


def expand_exact_colors(
    image_path, exact_colors, expansion_size=15, kernel_shape=cv2.MORPH_ELLIPSE
):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize an empty mask for the selected colors
    color_mask = np.zeros(image.shape[:2], dtype="uint8")

    # Loop over the exact colors to expand
    for color in exact_colors:
        # Ensure the color is in the correct format (np.uint8)
        lowerb = np.array([color["B"], color["G"], color["R"]], dtype=np.uint8)
        upperb = np.array([color["B"], color["G"], color["R"]], dtype=np.uint8)
        # Create a mask for the exact color
        mask = cv2.inRange(image, lowerb, upperb)

        # Combine the mask with the color mask
        color_mask = cv2.bitwise_or(color_mask, mask)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(kernel_shape, (expansion_size, expansion_size))

    # Dilate the mask to expand the selected colored regions
    expanded_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Apply the expanded mask to create the dilated regions on a black background
    expanded_image = cv2.bitwise_and(image, image, mask=expanded_mask)

    # Convert the expanded image to RGB and then to a PIL image
    expanded_image_rgb = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(expanded_image_rgb)

    return pil_image


def expand_colors_with_uniform_tolerance(
    image_path,
    exact_colors,
    tolerance=10,
    expansion_size=15,
    kernel_shape=cv2.MORPH_ELLIPSE,
):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize an empty mask for the selected colors
    color_mask = np.zeros(image.shape[:2], dtype="uint8")

    # Loop over the colors to expand
    for color in exact_colors:
        # Calculate the lower and upper bounds for the color with the same tolerance for all
        lowerb = np.array(
            [
                max(0, color["B"] - tolerance),
                max(0, color["G"] - tolerance),
                max(0, color["R"] - tolerance),
            ],
            dtype=np.uint8,
        )
        upperb = np.array(
            [
                min(255, color["B"] + tolerance),
                min(255, color["G"] + tolerance),
                min(255, color["R"] + tolerance),
            ],
            dtype=np.uint8,
        )

        # Create a mask for colors within the tolerance range
        mask = cv2.inRange(image, lowerb, upperb)

        # Combine the mask with the color mask
        color_mask = cv2.bitwise_or(color_mask, mask)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(kernel_shape, (expansion_size, expansion_size))

    # Dilate the mask to expand the selected colored regions
    expanded_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Apply the expanded mask to create the dilated regions on a black background
    expanded_image = cv2.bitwise_and(image, image, mask=expanded_mask)

    # Convert the expanded image to RGB and then to a PIL image
    expanded_image_rgb = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(expanded_image_rgb)

    return pil_image
