# Customized by Andrew Porter for ComfyUI
# 04/21/2024
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__
from .PettyPaint import PettyPaintComponent
from .PettyPaint import PettyPaintSDTurboScheduler
from .PettyPaintImageSave import PettyPaintImageSave
python = sys.executable


NODE_CLASS_MAPPINGS = {
    "PettyPaintComponent": PettyPaintComponent,
    "PettyPaintSDTurboScheduler": PettyPaintSDTurboScheduler,
    "PettyPaintImageSave": PettyPaintImageSave
}

print('\033[34mPetty Paint Custom Nodes: \033[92mLoaded\033[0m')