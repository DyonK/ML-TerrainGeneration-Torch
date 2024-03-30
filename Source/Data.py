import torch
import random

import numpy as np
import torchvision.transforms.functional as TF

from Source.Utils import CreateMapCropped,ShowMap,TanNormalize,WriteJson
from torch.utils.data import Dataset
from PIL import Image

class Map():
    def __init__(self,Path:str,Mode:str,Sample,Size:tuple) -> None:

        self.Path = Path
        self.Mode = Mode
        self.Sample = Sample
        self.Size = Size

        self.MapData = self.LoadMap()

    def ConvertOneHot(self,SegClassCount):

        FMap = torch.flatten(self.MapData,1,2)
        Unique , Indexed = torch.unique(FMap,return_inverse=True,dim=1)

        FoundLabels = Unique.shape[1]
        
        assert FoundLabels <= SegClassCount, "Segmentation classes found exeeded expected amount of classes."

        IndexedMap = torch.reshape(Indexed,(self.Size[1],self.Size[0]))
        OneHotMap = torch.nn.functional.one_hot(IndexedMap,SegClassCount).permute(2,0,1)

        return Unique , OneHotMap 
    
    def ConvertKoppen(self):

        assert self.Mode == 'RGBA'
        assert self.Sample == Image.Resampling.NEAREST

        KoppenDict = {
            "Water": [0.0,0.0,0.0,0.0],
            } #koppen rgb values and its climates

        #Convert to onehot

        
        #TODO: implement KoppenMap convertion to onehot
        pass

    def LoadMap(self) -> torch.tensor:
        Map = CreateMapCropped(self.Path,self.Size,self.Mode,Resample=self.Sample)
        Map = torch.tensor(Map).permute(2, 0, 1)
        return Map
    
    def ReturnMapdata(self) -> torch.tensor:
        return self.MapData
    
    def ReturnMapNumpy(self) -> np.array:
        return self.MapData.permute(1,2,0).numpy()

class MapDataSet(Dataset):
    def __init__(self,args,InputMap:Map,OutputMap:Map,SessionPath:str) -> None:

        self.DataDropRate = args.DataDropRate

        self.DatasetImageSize = args.DataImageSize
        self.ImageSize = args.ImageSize

        #get input map
        LabelList, self.InputMap = InputMap.ConvertOneHot(args.ConditionalClasses)

        #get output maps
        self.OutputMap = TanNormalize(OutputMap.ReturnMapdata())

        assert InputMap.Size == self.DatasetImageSize and OutputMap.Size == self.DatasetImageSize 

        #Write found LabelValues to Json
        WriteJson(SessionPath,"SegmentationClasses",torch.transpose(LabelList,0,1).tolist())

        self.PositionList, self.ListSize = self.GeneratePositionVector()
    
    def __len__(self):
        return self.ListSize
    
    def __getitem__(self,idx):

        Position = self.PositionList[idx]

        X = self.CropAndRoll(self.InputMap,self.ImageSize,Position)
        Y = self.CropAndRoll(self.OutputMap,self.ImageSize,Position)

        if random.random() > 0.5:
            X = TF.hflip(X)
            Y = TF.hflip(Y)

        if random.random() > 0.5:
            X = TF.vflip(X)
            Y = TF.vflip(Y)

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