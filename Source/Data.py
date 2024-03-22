import os
import torch
import random

import numpy as np
import torchvision.transforms.functional as TF

from Source.Utils import CreateMapCropped,ShowMap,TanNormalize,WriteJson
from torch.utils.data import Dataset

class MapDataSet(Dataset):
    def __init__(self,args,InputMap,OutputMap,SessionPath) -> None:

        self.DatasetImageSize = (args.DataImageSize, int( args.DataImageSize/2 ))
        self.ImageSize = (args.ImageSize, int( args.ImageSize/2 ))
        
        DataPath = os.path.join("",args.DataDir)

        #get input map
        self.InputMap = self.LoadMap(DataPath,InputMap)
        LabelList, self.InputMap = self.ConvertOneHot(self.InputMap,args.ConditionalClasses)

        #Write found LabelValues to Json
        WriteJson(SessionPath,"SegmentationClasses",torch.transpose(LabelList,0,1).tolist())
            
        #get output maps
        self.OutputMap = self.LoadMap(DataPath,OutputMap)
        self.OutputMap = TanNormalize(self.OutputMap)

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
    
    def ConvertOneHot(self,Map,SegClassCount):

        FMap = torch.flatten(Map,1,2)
        Unique , Indexed = torch.unique(FMap,return_inverse=True,dim=1)

        FoundLabels = Unique.shape[1]
        
        assert FoundLabels <= SegClassCount, "Segmentation classes found exeeded expected amount of classes."

        IndexedMap = torch.reshape(Indexed,self.DatasetImageSize)
        OneHotMap = torch.nn.functional.one_hot(IndexedMap,SegClassCount).permute(2, 0, 1)

        return Unique , OneHotMap

    def LoadMap(self,DataPath: str,MapArgs: tuple):
        MapName , MapMode , MapSample = MapArgs
        MapPath = os.path.join(DataPath,MapName)
        Map = CreateMapCropped(MapPath,self.DatasetImageSize,MapMode,Resample=MapSample)
        Map = torch.tensor(Map).permute(2, 0, 1)
        return Map