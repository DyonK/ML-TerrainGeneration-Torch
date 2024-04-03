import argparse
import datetime
import os

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
    Parser.add_argument('--ModelDir',           default='ModelLogs',       help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',          default='',                help='Name of trainingsession. If argument is empty a name will be generated')
    Parser.add_argument('--DataDir',            default='Data',            help='Which directory the training data is stored in')
    Parser.add_argument('--Device',             default='cuda',            help='which device to run project on')

    # learning args
    Parser.add_argument('--BatchSize',          default= 1,                type=int)
    Parser.add_argument('--LearningRate',       default= 3e-5,             type=float)
    Parser.add_argument('--TrainEpoch',         default= 15,                type=int)
    Parser.add_argument('--WeightDecay',        default= 0.00,             type=float)
    Parser.add_argument('--EmaRate',            default= 0.995,            type=float)
    Parser.add_argument('--DataDropRate',       default= 0.2,              type=float)
    Parser.add_argument('--DataImageSize',      default= (1024,512),        type=tuple)
    Parser.add_argument('--LogInterval',        default= 1024,             type=int)
    Parser.add_argument('--SaveInterval',       default= 1024 * 5,             type=int)

    # diffusion args
    Parser.add_argument('--TimeSteps',          default= 1000,             type=int)
    Parser.add_argument('--NoiseSchedule',      default= 'Linear',         type=str)

    #Model args
    Parser.add_argument('--ModelType',          default='Unet',            type=str)
    Parser.add_argument('--ImageSize',          default= (1024,512),        type=tuple)
    Parser.add_argument('--ConditionalClasses', default= 31,               type=int)
    Parser.add_argument('--TimeDims',           default= 256,              type=int)

    args = vars(Parser.parse_args())

    ModelTypes = ['Unet']

    if args['ModelType'] not in ModelTypes:
        raise ValueError("Please select valid modeltype: {}".format(ModelTypes))
    
    ScheduleTypes = ['Linear','Cosine']

    if args['NoiseSchedule'] not in ScheduleTypes:
        raise ValueError("Please select valid NoiseSchedule: {}".format(ModelTypes))
    
    CurrentPath , DataPath , OutputPath = PrepareFolders(args)

    WriteJson(CurrentPath,"Settings",args)

    #Data
    InputMap = Map(os.path.join(DataPath,"Köppen-Geiger_Climate_Classification_Map_(1980–2016)_no_borders.png"),'RGBA',Image.Resampling.NEAREST,args['DataImageSize'])
    OutputMap = Map(os.path.join(DataPath,"land_shallow_topo_2011_8192.jpg"),'RGB',Image.Resampling.LANCZOS,args['DataImageSize'])
    DataSet = MapDataSet(args,InputMap,OutputMap,CurrentPath)
    DataIterator = DataLoader(DataSet,batch_size=args['BatchSize'],shuffle=True)

    # Model and Diffusor
    if args['ModelType'] == 'Unet':
        Model = Unet(args).to(args['Device'])
    else:
        raise NotImplementedError()

    DiffusionObj = Diffusor(args,Device=args['Device'])

    #Training
    TrainObj = Trainer(args,Model,DiffusionObj,CurrentPath,Device=args['Device'])
    TrainObj.TrainLoop(DataIterator)

if __name__ == "__main__":
    main()