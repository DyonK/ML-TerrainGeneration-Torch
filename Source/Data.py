import torch
import random

import numpy as np
import torchvision.transforms.v2.functional as TF

from Source.Utils import CreateMapCropped,TanNormalize,ImgTorchToNumpy,TanReNormalize,ShowMap
from torch.utils.data import Dataset
from PIL import Image

# KOPPEN CLASSES BY COLOUR
KoppenMask = [
  (0.0,0.0,0.0,0.0),
  (0.0,0.0,1.0,1.0),
  (0.0,0.27450981736183167,0.37254902720451355,1.0),
  (0.0,0.47058823704719543,1.0,1.0),
  (0.0,0.4901960790157318,0.4901960790157318,1.0),
  (0.0,1.0,1.0,1.0),
  (0.19607843458652496,0.0,0.529411792755127,1.0),
  (0.19607843458652496,0.5882353186607361,0.19607843458652496,1.0),
  (0.19607843458652496,0.7843137383460999,0.0,1.0),
  (0.21568627655506134,0.7843137383460999,1.0,1.0),
  (0.27450981736183167,0.6666666865348816,0.9803921580314636,1.0),
  (0.29411765933036804,0.3137255012989044,0.7019608020782471,1.0),
  (0.3490196168422699,0.47058823704719543,0.8627451062202454,1.0),
  (0.3921568691730499,0.7843137383460999,0.3921568691730499,1.0),
  (0.3921568691730499,1.0,0.3137255012989044,1.0),
  (0.4000000059604645,0.4000000059604645,0.4000000059604645,1.0),
  (0.5882353186607361,0.19607843458652496,0.5882353186607361,1.0),
  (0.5882353186607361,0.3921568691730499,0.5882353186607361,1.0),
  (0.5882353186607361,0.5882353186607361,0.0,1.0),
  (0.5882353186607361,1.0,0.5882353186607361,1.0),
  (0.6666666865348816,0.686274528503418,1.0,1.0),
  (0.6980392336845398,0.6980392336845398,0.6980392336845398,1.0),
  (0.7843137383460999,0.0,0.7843137383460999,1.0),
  (0.7843137383460999,0.7843137383460999,0.0,1.0),
  (0.7843137383460999,1.0,0.3137255012989044,1.0),
  (0.9607843160629272,0.6470588445663452,0.0,1.0),
  (1.0,0.0,0.0,1.0),
  (1.0,0.0,1.0,1.0),
  (1.0,0.5882353186607361,0.5882353186607361,1.0),
  (1.0,0.8627451062202454,0.3921568691730499,1.0),
  (1.0,1.0,0.0,1.0)
]

class Map():
    def __init__(self,Path:str,Mode:str,Sample,Size:tuple) -> None:

        self.Path = Path
        self.Mode = Mode
        self.Sample = Sample
        self.Size = Size

        self.MapData = self.LoadMap()
    
    def ConvertKoppen(self) -> torch.tensor:

        assert self.Mode == 'RGBA'
        assert self.Sample == Image.Resampling.NEAREST

        Mapping = {tuple(c): t for c, t in zip(KoppenMask, range(len(KoppenMask)))}

        Mask = torch.zeros((self.Size[1],self.Size[0]),dtype=torch.int64)

        for Colour in Mapping:

            IdMap = (self.MapData == torch.tensor(Colour,dtype=torch.float32)[:,None,None])
            validx = (IdMap.sum(0)==4) 
            Mask[validx] = Mapping[Colour]
        
        OneHotMap = torch.nn.functional.one_hot(Mask,len(KoppenMask)).permute(2,0,1)
        return OneHotMap
        
    def LoadMap(self) -> torch.tensor:
        Map = CreateMapCropped(self.Path,self.Size,self.Mode,Resample=self.Sample)
        Map = torch.tensor(Map).permute(2, 0, 1)
        return Map
    
    def ReturnMapdata(self) -> torch.tensor:
        return self.MapData
    
    def ReturnMapNumpy(self) -> np.array:
        return ImgTorchToNumpy(self.MapData)

class MapDataSet(Dataset):
    def __init__(self,args,InputMap:Map,OutputMap:Map) -> None:

        self.DataDropRate = args['DataDropRate']

        self.DatasetImageSize = args['DataImageSize']
        self.ImageSize = args['ImageSize']

        #get input map
        self.InputMap = InputMap.ConvertKoppen()

        #get output maps
        self.OutputMap = TanNormalize(OutputMap.ReturnMapdata())

        assert InputMap.Size == self.DatasetImageSize and OutputMap.Size == self.DatasetImageSize 

        self.PositionList, self.ListSize = self.GeneratePositionVector()
    
    def __len__(self):
        return self.ListSize
    
    def __getitem__(self,idx):

        Position = self.PositionList[idx]

        Input = self.InputMap
        Output = self.OutputMap

        #Flip data maps for improved zoom sampling
        Input,Output = self.RandomFlip(Input,Output)

        ZoomScale = random.random()
        X = self.CropRollZoom(Input,self.ImageSize,Position,ZoomScale,Interpolation=TF.InterpolationMode.NEAREST)
        Y = self.CropRollZoom(Output,self.ImageSize,Position,ZoomScale,Interpolation=TF.InterpolationMode.NEAREST)

        X,Y = self.RandomFlip(X,Y)
        
        if not random.random() > self.DataDropRate:
            X = torch.zeros_like(X)

        return X,Y

    #create [N,2] np array with all possible pixel positions
    def GeneratePositionVector(self) -> np.array:

        #horizontal dimention can be wrapped
        HorDim = self.DatasetImageSize[0]
        VerDim = max(self.DatasetImageSize[1]- self.ImageSize[1],1)

        NSize = (HorDim * VerDim)

        Indices = np.stack(np.indices((HorDim,VerDim)))
        Indices = np.transpose(Indices)

        Indices = Indices.reshape(NSize,2)

        return Indices, NSize
    
    #Custom Crop function that wraps horizontal dim
    def CropAndRoll(self,Map,Size,Position):

        Y,X = Position

        return torch.roll(Map,-Y,dims=2)[:,X:X+Size[1],0:Size[0]]
    
    #Zoom function based on wrapping crop
    def CropRollZoom(self,Map,Size,Position,ZoomScale,Interpolation):

        Y,X = Position
        MaxY,MaxX = self.DatasetImageSize
        MinY,MinX = Size

        MaxX -= X
        MaxX = max(MaxX,MinX)
        MaxLocalScale = MaxX / MinX
        
        LocalScale = 1.0 + ZoomScale * (MaxLocalScale - 1.0)
        
        ZoomY = int(LocalScale * MinY)
        ZoomX = int(LocalScale * MinX)

        ZoomSize = (ZoomY,ZoomX)

        Map = self.CropAndRoll(Map,ZoomSize,Position)
        Map = TF.resize(Map,(Size[1],Size[0]),interpolation=Interpolation)

        return Map
    
    def RandomFlip(self,Map1,Map2):

        if random.random() > 0.5: 
            Map1 = TF.hflip(Map1)
            Map2 = TF.hflip(Map2)

        if random.random() > 0.5:
            Map1 = TF.vflip(Map1)
            Map2 = TF.vflip(Map2)

        return Map1,Map2




