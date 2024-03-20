import json
import os

import numpy as np

from PIL import Image


def WriteJson(Path,Name,Data):
    Path = os.path.abspath(Path)
    with open(os.path.join(Path,Name + ".json"),'W') as f:
        json.dump(Data,f,indent=2)

def CreateMap(Path,Mode):
    Map = Image.open(Path).convert(Mode)
    Map = np.array(Map).astype(np.float32)
    Map /= np.max(Map)
    return Map

def CreateMapCropped(Path,Size,Mode,Resample=Image.NEAREST):
    Map = Image.open(Path).convert(Mode).resize(Size,resample=Resample)
    Map = np.array(Map).astype(np.float32)
    Map /= np.max(Map)
    return Map

def ShowMap(Map):
    Img = Image.fromarray((Map * 255).astype(np.uint8))
    Img.show()