import os
import argparse

from Source.Utils import *

import numpy as np

def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Tolerance',    default= 0.06,  type=float)
    Parser.add_argument('--KoppenPath',   default="Data/Köppen-Geiger_Climate_Classification_Map_(1980–2016)_no_borders.png" , type=str)
    Parser.add_argument('--OutputPath',   default="Data/land_shallow_topo_8192.tif" , type=str)
    args = vars(Parser.parse_args())

    InputMap = CreateMap(args['KoppenPath'],'RGBA')

    Size = (InputMap.shape[1],InputMap.shape[0])
    OutputMap = CreateMapCropped(args['OutputPath'],Size,'RGB',Image.Resampling.LANCZOS)

    #InputLakes into Koppenmap
    WaterPixel = OutputMap[0][0]
    DiffMap = (OutputMap - WaterPixel) < args['Tolerance']
    Mask = np.all(DiffMap ,axis=-1)
    InputMap[Mask] = [0.,0.,0.,0.]

    #Make all water the same colour
    Mask = (InputMap[:,:,3] == 0.)
    InputMap[Mask] = [0.,0.,0.,0.]
    OutputMap[Mask] = WaterPixel

    #Save
    os.makedirs("Data",exist_ok=True)
    SaveMap(InputMap,"Data/InputMap.png")
    SaveMap(OutputMap,"Data/OutputMap.png")


if __name__ == "__main__":
    main()