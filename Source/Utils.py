import json
import os

import numpy as np

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000

def WriteJson(Path: str, Name: str ,Data: dict):
    Path = os.path.abspath(Path)
    with open(os.path.join(Path,Name + ".json"),'w') as f:
        json.dump(Data,f,indent=2)

def CreateMap(Path: str, Mode) -> np.array:
    Map = Image.open(Path).convert(Mode)
    Map = np.array(Map).astype(np.float32)
    Map /= np.max(Map)
    return Map

def CreateMapCropped(Path: str, Size: tuple, Mode: str, Resample=Image.Resampling.NEAREST) -> np.array:
    Map = Image.open(Path).convert(Mode).resize(Size,resample=Resample)
    Map = np.array(Map).astype(np.float32)
    Map /= np.max(Map)
    return Map

def ShowMap(Map: np.array):
    Img = Image.fromarray((Map * 255).astype(np.uint8))
    Img.show()

def TanNormalize(X):
    return (X-0.5)/0.5

def TanReNormalize(X):
    return (X+1)/2.0