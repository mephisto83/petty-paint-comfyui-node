# Customized by Andrew Porter for ComfyUI
# 04/21/2024
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__
from .PettyPaint import PettyPaintConnector
python = sys.executable



def is_installed(package, package_overwrite=None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    if spec is None:
        print(f"Installing {package}...")
        command = f'"{python}" -m pip install {package}'
  
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)

        if result.returncode != 0:
            print(f"Couldn't install\nCommand: {command}\nError code: {result.returncode}")


NODE_CLASS_MAPPINGS = {
    "MultiAreaConditioningAP": MultiAreaConditioningAP,
    "CR Load LoRA": CR_LoraLoader_AP,
    "CR LoRA Stack":CR_LoRAStack_AP,
    "CR Apply LoRA Stack":CR_ApplyLoRAStack_AP,
    "CR Random LoRA Stack":CR_RandomLoRAStack_AP,
    "CR Random Weight LoRA":CR_RandomWeightLoRA_AP,    
    "Modal Switch": MODEL_SWITCH,
    "Modal Loader": APCheckpointLoaderSimple
}

print('\033[34mAporter Custom Nodes: \033[92mLoaded\033[0m')