import os
import argparse

from Source.Utils import *

import numpy as np

def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Tolerance',    default= 0.05,  type=float)
    args = vars(Parser.parse_args())

    InputMap = CreateMap("Data/Köppen-Geiger_Climate_Classification_Map_(1980–2016)_no_borders.png",'RGBA')

    Size = (InputMap.shape[1],InputMap.shape[0])
    OutputMap = CreateMapCropped("Data/land_shallow_topo_2011_8192.jpg",Size,'RGB',Image.Resampling.LANCZOS)

    #InputLakes into Koppenmap
    WaterPixel = OutputMap[0][0]
    DiffMap = (OutputMap - WaterPixel) < args['Tolerance']
    Mask = np.all(DiffMap ,axis=-1)
    InputMap[Mask] = [0.,0.,0.,0.]

    #Make all water the same colour
    Mask = np.all(InputMap == [0.,0.,0.,0.],axis=-1)
    OutputMap[Mask] = WaterPixel

    #Save
    SaveMap(InputMap,"Data/InputMap.png")
    SaveMap(OutputMap,"Data/OutputMap.png")


if __name__ == "__main__":
    main()