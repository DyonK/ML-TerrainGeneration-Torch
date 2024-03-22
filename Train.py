import argparse
import datetime
import os

from Source.Utils import WriteJson,ShowMap,TanReNormalize
from Source.Data import MapDataSet
from Source.Trainer import Trainer
from PIL import Image
from torch.utils.data import DataLoader

def PrepareFolders(args):

    BaseDir = ""

    #make modellogs folder
    LogPath = os.path.join(BaseDir,args.ModelDir)
    os.makedirs(LogPath,exist_ok=True)

    #make output folder
    OutputPath = os.path.join(BaseDir,"Output")
    os.makedirs(OutputPath,exist_ok=True)

    #make data folder
    DataPath = os.path.join(BaseDir,args.DataDir)
    os.makedirs(DataPath,exist_ok=True)

    #make folder for trainingsession
    if args.ModelName == '':
        SessionName = GenerateModelName()
    else:
        SessionName = args.ModelName
        
    SessionPath = os.path.join(LogPath,SessionName)

    try:
        os.makedirs(SessionPath,exist_ok=False)
    except Exception as e:
        print(e)

    return SessionPath


def GenerateModelName():

    DateTime= datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    Name = 'Session-' + DateTime
    return Name


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelDir',           default='ModelLogs',       help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',          default='',                help='Name of trainingsession. If argument is empty a name will be generated')
    Parser.add_argument('--ModelType',          default='Unet',            help='Which model architecture to run')
    Parser.add_argument('--DataDir',            default='Data',            help='Which directory the training data is stored in')

    # learning args
    Parser.add_argument('--BatchSize',          default= 1,                type=int)
    Parser.add_argument('--LearningRate',       default= 1e-4,             type=float)
    Parser.add_argument('--TrainEpoch',         default= 100,              type=int)
    Parser.add_argument('--WeightDecay',        default= 0.0,              type=float)
    Parser.add_argument('--EmaRate',            default= 0.9999,           type=float)
    Parser.add_argument('--DataDropRate',       default= 0.2,              type=float)
    Parser.add_argument('--DataImageSize',      default= 4320,             type=int)
    Parser.add_argument('--LogInterval',        default= 100,              type=int)
    Parser.add_argument('--SaveInterval',       default= 1000,             type=int)

    # diffusion args
    Parser.add_argument('--TimeSteps',          default= 1000,             type=int)
    Parser.add_argument('--NoiseSchedule',      default= 'Linear',         type=str)

    #Model args
    Parser.add_argument('--ImageSize',          default= 4320,               type=int)
    Parser.add_argument('--ConditionalClasses', default= 31,               type=int)

    args = Parser.parse_args()

    ModelTypes = ['Unet']

    if args.ModelType not in ModelTypes:
        raise ValueError("Please select valid modeltype: {}".format(ModelTypes))
    
    ScheduleTypes = ['Linear']

    if args.NoiseSchedule not in ScheduleTypes:
        raise ValueError("Please select valid NoiseSchedule: {}".format(ModelTypes))
    
    CurrentPath = PrepareFolders(args)

    WriteJson(CurrentPath,"Settings",vars(args))

    #TODO: create logger

    InputMap = ("Köppen-Geiger_Climate_Classification_Map_(1980–2016)_no_borders.png",'RGBA',Image.NEAREST)
    OutputMap = ("land_shallow_topo_2011_8192.jpg",'RGB',Image.LANCZOS)
    DataSet = MapDataSet(args,InputMap,OutputMap,CurrentPath)
    DataIterator = DataLoader(DataSet,batch_size=args.BatchSize,shuffle=True)

    

    #TODO: call training loop here
    TrainObj = Trainer()







if __name__ == "__main__":
    main()