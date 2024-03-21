import os

import numpy as np

from Source.Utils import CreateMapCropped,ShowMap

class DataLoader():
    def __init__(self,args,InputMaps,OutputMap) -> None:

        self.DatasetImageSize = (args.DataImageSize, int( args.DataImageSize/2 ))
        self.ImageSize = (args.ImageSize, int( args.ImageSize/2 ))
        
        DataPath = os.path.join("",args.DataDir)

        #get input maps
        self.InputMaps = []
        for MapName,MapMode,MapSample in InputMaps:

            MapPath = os.path.join(DataPath,MapName)
            Map = CreateMapCropped(MapPath,self.ImageSize,MapMode,Resample=MapSample)
            self.InputMaps.append(Map)

        
        #get output maps
        OutputMapName , OutputMapMode , OutputMapSample = OutputMap
        MapPath = os.path.join(DataPath,OutputMapName)
        self.OutputMap = CreateMapCropped(MapPath,self.ImageSize,OutputMapMode,OutputMapSample)

        PositionVector = self.GeneratePositionVector()


    def GeneratePositionVector(self) -> np.array:

        HorDim = self.DatasetImageSize[0]
        VerDim = max(self.DatasetImageSize[1]- self.ImageSize[1],1)

        Indices = np.stack(np.indices((HorDim,VerDim)))
        


        pass

#TODO: add roll and crop functions
#TODO: find out how pytorch wants dataloading goes