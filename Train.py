import argparse
import datetime
import os

import torch.utils

from Source.Utils import WriteJson
from Source.Data import *
from Source.Trainer import Trainer
from Source.Unet import Unet
from Source.Diffusion import Diffusor
from PIL import Image
from torch.utils.data import DataLoader

def PrepareFolders(args):

    BaseDir = ""

    #make modellogs folder
    LogPath = os.path.join(BaseDir,args['ModelDir'])
    os.makedirs(LogPath,exist_ok=True)

    #make input folder
    OutputPath = os.path.join(BaseDir,"Input")
    os.makedirs(OutputPath,exist_ok=True)

    #make output folder
    OutputPath = os.path.join(BaseDir,"Output")
    os.makedirs(OutputPath,exist_ok=True)

    #make data folder
    DataPath = os.path.join(BaseDir,args['DataDir'])
    os.makedirs(DataPath,exist_ok=True)

    #make folder for trainingsession
    if args['ModelName'] == '':
        SessionName = GenerateModelName()
    else:
        SessionName = args['ModelName']
        
    SessionPath = os.path.join(LogPath,SessionName)

    try:
        os.makedirs(SessionPath,exist_ok=False)
    except Exception as e:
        print(e)

    return SessionPath , DataPath , OutputPath

def GenerateModelName():

    DateTime= datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    Name = 'Session-' + DateTime
    return Name

def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelDir',           default= 'ModelLogs',       help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',          default= '',                help='Name of trainingsession. If argument is empty a name will be generated')
    Parser.add_argument('--DataDir',            default= 'Data',            help='Which directory the training data is stored in')
    Parser.add_argument('--Device',             default= 'cuda',            help='which device to run project on')

    # diffusion args 
    Parser.add_argument('--BatchSize',          default= 2,                type=int)
    Parser.add_argument('--LearningRate',       default= 2e-4,             type=float)
    Parser.add_argument('--LRDecay',            default= 1.0,              type=float)
    Parser.add_argument('--Lambda',             default= 0.001,            type=float)
    Parser.add_argument('--TrainEpoch',         default= 1,              type=int)
    Parser.add_argument('--WeightDecay',        default= 0.00,             type=float)
    Parser.add_argument('--EmaRate',            default= 0.9999,           type=float)
    Parser.add_argument('--DataDropRate',       default= 0.2,              type=float)
    Parser.add_argument('--DataDropDelay',      default= 0,                type=int)
    Parser.add_argument('--LogInterval',        default= 1024,             type=int)
    Parser.add_argument('--SaveInterval',       default= 5000,            type=int)

    # Data args
    Parser.add_argument('--DataImageSize',      default= (1024,512),       type=tuple)
    Parser.add_argument('--DataZooming',        default= True,             type=bool)
    Parser.add_argument('--MaxZooming',         default= 0.2,             type=float)
    Parser.add_argument('--DataSpacing',        default= False,             type=bool)
    Parser.add_argument('--DataWorkers',        default= 16,               type=int)
    Parser.add_argument('--ValSplit',           default= 0.0,              type=float)

    # diffusion args
    Parser.add_argument('--TimeSteps',          default= 1000,             type=int)
    Parser.add_argument('--NoiseSchedule',      default= 'Linear',         type=str)

    # Model args
    Parser.add_argument('--ModelType',           default= 'Unet',           type=str)
    Parser.add_argument('--ImageSize',           default= (512,256),        type=tuple)
    Parser.add_argument('--ConditionalClasses',  default= 31,               type=int)
    Parser.add_argument('--TimeDims',            default= 256,              type=int)
    Parser.add_argument('--Channels',            default= [32,64,64,128,128,128,256], type=list)
    Parser.add_argument('--ResBlocks',           default = 2,               type=int)
    Parser.add_argument('--AttentionThreshhold', default = 4,               type=int)

    args = vars(Parser.parse_args())
    
    CurrentPath , DataPath , OutputPath = PrepareFolders(args)

    WriteJson(CurrentPath,"Settings",args)

    #Data
    InputMap = Map(os.path.join(DataPath,"InputMap.png"),'RGBA',Image.Resampling.NEAREST,args['DataImageSize'])
    OutputMap = Map(os.path.join(DataPath,"OutputMap.png"),'RGB',Image.Resampling.LANCZOS,args['DataImageSize'])
    DataSet = MapDataSet(args,InputMap,OutputMap)

    if args['ValSplit'] > 0.0:
        TrainSet , ValSet = torch.utils.data.random_split(DataSet,[1 - args['ValSplit'],args['ValSplit']])
        DataIterator = DataLoader(TrainSet,batch_size=args['BatchSize'],shuffle=True,num_workers=args['DataWorkers'],pin_memory=True,drop_last=True,persistent_workers=True)
        ValIterator = DataLoader(ValSet,batch_size=args['BatchSize'],shuffle=True,drop_last=True)
    else:
        DataIterator = DataLoader(DataSet,batch_size=args['BatchSize'],shuffle=True,num_workers=args['DataWorkers'],pin_memory=True,drop_last=True,persistent_workers=True)
        ValIterator = None
 
    # Model and Diffusor
    if args['ModelType'] == 'Unet':
        Model = Unet(args).to(args['Device'])
    else:
        raise NotImplementedError()

    DiffusionObj = Diffusor(args,Device=args['Device'])

    #Training
    TrainObj = Trainer(args,Model,DiffusionObj,CurrentPath,Device=args['Device'])
    TrainObj.TrainLoop(DataIterator,ValIterator)

if __name__ == "__main__":
    main()