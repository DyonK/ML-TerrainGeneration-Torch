import argparse
import os
import torch

from Source.Utils import LoadJson,SaveMap
from Source.Diffusion import Diffusor
from Source.Unet import Unet
from Source.Data import Map
from PIL import Image

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelDir',           default='ModelLogs',                help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',          default='Session-2024-04-03--15-35-26',                         help='Name of session to use for sampling')
    Parser.add_argument('--ModelVersion',       default='Model_Ema_Final.ckpt.pt',  help='Which intermedial model to load. Defaults to final')
    Parser.add_argument('--Device',             default='cuda',                     help='which device to run project on')

    Parser.add_argument('--CFGScale',           default=3.0,                        help='What CFGScale to use when sampling')
    args = vars(Parser.parse_args())

    SessionPath = os.path.join("",args['ModelDir'],args['ModelName'])
    InputDir = os.path.join("",'Input')
    OutDir = os.path.join("",'Output')

    if os.path.exists(SessionPath) == False or args['ModelName'] == '':
        raise ValueError("Model does not exist at {}".format(SessionPath))
    
    ModelSettings = LoadJson(os.path.join(SessionPath,'Settings.json'))

    if ModelSettings['ModelType'] == 'Unet':
        Model = Unet(ModelSettings).to(args['Device'])

    WeightsPath = os.path.join(SessionPath,'Models',args['ModelVersion'])
    if  os.path.exists(WeightsPath) == True:
        Model.load_state_dict(torch.load(WeightsPath))
    else:
        raise ValueError("Weights do not exist at {}".format(WeightsPath))

    DiffusionObj = Diffusor(ModelSettings,Device=args['Device'])

    FileList = os.listdir(InputDir)
    for File in FileList:
        
        InputPath = os.path.join(InputDir,File)

        if os.path.isfile(InputPath):
            InputMap = Map(InputPath,'RGBA',Image.Resampling.NEAREST,ModelSettings['ImageSize'])
            InputMap = InputMap.ConvertKoppen()[None,...]

            OutImage = DiffusionObj.SampleImage(Model,1,InputMap,args['CFGScale'])
            OutImage = OutImage.to('cpu')[0,...].permute(1,2,0).numpy()
            SaveMap(OutImage,os.path.join(OutDir,File))

            print("Sampling {} has been completed".format(File))
    
    
if __name__ == "__main__":
    main()