import json
import os
import torch
import torchvision

import numpy as np

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000

def WriteJson(Path: str, Name: str ,Data: dict)->None:
    Path = os.path.abspath(Path)
    with open(os.path.join(Path,Name + ".json"),'w') as f:
        json.dump(Data,f,indent=2)

def LoadJson(Path:str) -> dict:
    Path = os.path.abspath(Path)
    with open(Path,'r') as f:
        return json.load(f)

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

def ShowMap(Map: np.array) -> None:
    Img = NumpyToPil(Map)
    Img.show()

def SaveMap(Map: np.array, Path: str) -> None:
    img = NumpyToPil(Map)
    img.save(Path)

def TanNormalize(X):
    return (X-0.5)/0.5

def TanReNormalize(X):
    return (X+1)/2.0

def ImgTorchToNumpy(X) -> np.array:
    return X.permute(1,2,0).numpy()

def NumpyToPil(X) -> Image.Image:
    return Image.fromarray((X * 255).astype(np.uint8))

def SaveVideo(Path:str, X:list) ->None:

    VideoTensor = torch.squeeze(X)
    VideoTensor = VideoTensor.permute(0,2,3,1)
    VideoTensor = (VideoTensor * 255).to(dtype=torch.uint8)

    torchvision.io.write_video(Path,VideoTensor,30,video_codec='libx264',options={"b":"10000000"})