import argparse
import datetime
import os

from Source.Defaults import HparamDefaults,DiffDefaults,ModelDefaults
from Source.Utils import CreateMap,CreateMapCropped,ShowMap

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
    Parser.add_argument('--ModelDir',       default='ModelLogs',       help='Folder in which trained models are stored')
    Parser.add_argument('--ModelName',      default='',                help='Name of trainingsession. If argument is empty a name will be generated')
    Parser.add_argument('--ModelType',      default='Unet',            help='Which model architecture to run')
    Parser.add_argument('--DataDir',        default='Data',            help='Which directory the training data is stored in')



    args = Parser.parse_args()

    ModelTypes = ['Unet']

    if args.ModelType not in ModelTypes:
        raise ValueError("Please select valid modeltype: {}".format(ModelTypes))
    
    CurrentPath = PrepareFolders(args)

    Hparams = HparamDefaults()
    DiffusionSettings = DiffDefaults()
    ModelSettings = ModelDefaults()

    #TODO: create logger

    #TODO: load in dataset

    #TODO: call training loop here

    # #test:
    # map = CreateMap("Data/Köppen-Geiger_Climate_Classification_Map_(1980–2016)_no_borders.png",'RGBA')
    # ShowMap(map)

    






if __name__ == "__main__":
    main()