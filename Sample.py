import argparse
import os
import torch

from Source.Utils import LoadJson,SaveMap,ImgTorchToNumpy,SaveVideo
from Source.Diffusion import Diffusor
from Source.Unet import Unet
from Source.Data import Map
from PIL import Image

def SaveVideos(FrameList,Folder:str,Name:str)->None:

    FrameList = torch.stack(FrameList).permute(1,0,2,3,4)

    for i in range(FrameList.shape[0]):
        OutFrames = FrameList[i,...]
        SaveVideo(os.path.join(Folder,"{}.mp4".format(Name + "_" + str(i))),OutFrames)

def SaveImages(ImageList,Folder:str,Name:str)->None:
    for i in range(ImageList.shape[0]):
        OutImage = ImgTorchToNumpy(ImageList[i,...])
        SaveMap(OutImage,os.path.join(Folder,Name + "_" + str(i)+ ".png"))

def PrepareKoppenMap(Path:str,Size:tuple,Batches:int):
    InputMap = Map(Path,'RGBA',Image.Resampling.NEAREST,Size)
    InputMap = InputMap.ConvertKoppen()[None,...]

    InputMap = InputMap.repeat((Batches,1,1,1))

    return InputMap

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelDir',           default='ModelLogs',                              help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',          default='Session-2024-05-28--13-03-45',           help='Name of session to use for sampling')
    Parser.add_argument('--ModelVersion',       default='Model_Ema_Final.ckpt.pt',                help='Which intermedial model to load. Defaults to final')
    Parser.add_argument('--Device',             default='cuda',                                   help='which device to run project on')

    Parser.add_argument('--CFGScale',           default= 3.0,                       type = float, help='What CFGScale to use when sampling')
    Parser.add_argument('--SaveIntermediate',   default= False,                     type = bool,  help='Save intermediate steps into output folder')
    Parser.add_argument('--SaveVideo',          default= True,                      type = bool,  help='Save intermediate steps into  video format')
    Parser.add_argument('--ImageSize',          default= (512,256),                 type = tuple)
    Parser.add_argument('--Samples',            default= 1,                         type = int)
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
        
        FileName = File.split(".")[0]
        InputPath = os.path.join(InputDir,File)
        
        if os.path.isfile(InputPath):

            OutputFolder = os.path.join(OutDir,FileName)
            os.makedirs(OutputFolder,exist_ok=True)
            
            InputMap = PrepareKoppenMap(InputPath,args['ImageSize'],args['Samples'])

            OutImage, ProcessList = DiffusionObj.SampleImage(Model,args['Samples'],InputMap,args['CFGScale'],ReturnSteps=True,ImageSize=args['ImageSize'])
            SaveImages(OutImage,OutputFolder,FileName)

            if args['SaveIntermediate']:
                for i,Item in enumerate(ProcessList):
                    SaveImages(Item,OutputFolder,str(i))

            if args['SaveVideo']:
                SaveVideos(ProcessList,OutputFolder,FileName)

            print("Sampling {} has been completed".format(File))
    
    
if __name__ == "__main__":
    main()