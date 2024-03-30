import torch
import time
import copy
import os
import logging

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Source.Diffusion import Diffusor
from Source.Utils import ShowMap,SaveMap
from typing import Optional
from torchvision.utils import make_grid


import numpy as np

class Trainer():
    def __init__(self,args,Model : torch.nn.Module, Diffusor : Diffusor, SessionPath: Optional[str] = None) -> None:

        self.Epoch = args.TrainEpoch
        self.LogInterval = args.LogInterval
        self.SaveInterval = args.SaveInterval
        self.LearningRate = args.LearningRate
        self.WeightDecay = args.WeightDecay
        self.BatchSize = args.BatchSize
        self.EmaRate = args.EmaRate
        self.Device = args.Device

        if SessionPath != None:
            #Data Logging

            self.SessionPath = SessionPath

            self.SavePath = os.path.join(self.SessionPath,'Models')
            os.mkdir(self.SavePath)

            self.SamplePath = os.path.join(self.SessionPath,'TrainingSamples')
            os.mkdir(self.SamplePath)

            self.Board = SummaryWriter(self.SessionPath)

            LoggingPath = os.path.join(self.SessionPath,'TrainLog.log')
            self.Logger = logging.getLogger(__name__)
            logging.basicConfig(filename=LoggingPath,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

        else:
            self.SessionPath = None

        self.Model = Model
        self.Diffusor = Diffusor

        self.Optimizer = torch.optim.AdamW(Model.parameters(),lr=self.LearningRate,weight_decay=self.WeightDecay)
        self.Criterion = torch.nn.MSELoss()

        self.EmaModel = copy.deepcopy(Model).eval().requires_grad_(False)

    def TrainLoop(self,Dataset: DataLoader) -> None:

        Time1 = time.time()
        DataSetLength = len(Dataset)

        LossList = []

        for Epoch in range(0,self.Epoch):

            for i,(Cond,OutImage) in enumerate(Dataset,0):

                Step = Epoch * DataSetLength + i

                Cond = Cond.to(self.Device)
                OutImage = OutImage.to(self.Device)

                T = self.Diffusor.RandomTimeStep(self.BatchSize).to(self.Device)
                X,Noise = self.Diffusor.AddNoise(OutImage,T)
                
                Loss = self.TrainStep(X,Noise,Cond,T)
                LossList.append(Loss)

                if self.SessionPath != None:
                        self.Board.add_scalar("MSE", Loss, global_step=Step)
                        self.Logger.info("MSE:{} | Step:{} | Epoch:{} | Itteration:{}".format(Loss,Step,Epoch,i))

                if i % self.LogInterval == 0:

                    AveLoss = np.mean(LossList)

                    Time2 = time.time()
                    Msg = 'Epoch: {} | Itteration: {} | Step: {}| average diffusion loss: {} | time(s):{} '.format(Epoch,i,Step,AveLoss,Time2-Time1)
                    print(Msg)
                    self.Logger.info(Msg)
                    Time1 = time.time()

                    LossList.clear()

                if Step % self.SaveInterval == 0 and self.SessionPath != None:
                    
                    self.SaveModel(str(Step))

                    SampleCond = next(iter(Dataset))[0]
                    ModelGrid = self.SampleLog(self.Model,self.BatchSize,SampleCond,"Model_{}".format(Step))
                    EmaGrid = self.SampleLog(self.EmaModel,self.BatchSize,SampleCond,"EmaModel_{}".format(Step))

                    self.Board.add_image("ModelTrainSamples",ModelGrid,dataformats='HWC',global_step=Step)
                    self.Board.add_image("EmaModelTrainSamples",EmaGrid,dataformats='HWC',global_step=Step)
        
        if self.SavePath != None:
            self.SaveModel('Final')
            self.Board.close()

            SampleCond = next(iter(Dataset))[0]
            ModelGrid = self.SampleLog(self.Model,self.BatchSize,SampleCond,"Model_Final")
            EmaGrid = self.SampleLog(self.EmaModel,self.BatchSize,SampleCond,"EmaModel_Final")

            self.Board.add_image("ModelTrainSamples",ModelGrid,dataformats='HWC',global_step=Step)
            self.Board.add_image("EmaModelTrainSamples",EmaGrid,dataformats='HWC',global_step=Step)

    def TrainStep(self,X,Y,Cond,T) -> float:

        Output = self.Model(X,T,Cond)

        Loss = self.Criterion(Output,Y)

        self.Optimizer.zero_grad()
        Loss.backward()
        self.Optimizer.step()

        #Exponential moving average model update
        for Weight,EmaWeight in zip(self.Model.parameters(),self.EmaModel.parameters()):
            EmaWeight.data = self.EmaRate * EmaWeight.data + (1.0-self.EmaRate) * Weight.data

        return Loss.item()
    
    def ReturnEmaModel(self):
        return self.EmaModel
    
    def SampleLog(self,Model,N,Cond, Indicator:str):

        assert self.SessionPath != None

        self.Logger.info('Start Sampling Images for Logging')

        SampledImages = self.Diffusor.SampleImage(Model,N,Cond).to('cpu')

        SampleGrid = make_grid(SampledImages).permute(1,2,0).numpy()

        SamplePath = os.path.join(self.SamplePath,'{}.png'.format(Indicator))
        SaveMap(SampleGrid,SamplePath)

        self.Logger.info('Sampling Images for Logging Complete')

        return SampleGrid
    
    def SaveModel(self,Indicator):

        assert self.SessionPath != None

        self.Logger.info('Start Saving Modelstates')
        
        ModelPath = os.path.join(self.SavePath,'Model_{}.ckpt.pt'.format(Indicator))
        torch.save(self.Model.state_dict(),ModelPath)

        EmaPath = os.path.join(self.SavePath,'Model_Ema_{}.ckpt.pt'.format(Indicator))
        torch.save(self.EmaModel.state_dict(),EmaPath)

        OptPath = os.path.join(self.SavePath,'Optim_{}.ckpt.pt'.format(Indicator))
        torch.save(self.Optimizer.state_dict(),OptPath)