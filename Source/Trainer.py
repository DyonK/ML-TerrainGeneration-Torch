import torch
import time
import copy
import os
import logging
import random
import math

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Source.Diffusion import Diffusor
from Source.Utils import ShowMap,SaveMap,ImgTorchToNumpy
from typing import Optional
from torchvision.utils import make_grid

import numpy as np

class Trainer():
    def __init__(self,args,Model : torch.nn.Module, Diffusor : Diffusor, SessionPath: Optional[str] = None , Device = 'cpu') -> None:

        self.Epoch = args['TrainEpoch']
        self.LogInterval = args['LogInterval']
        self.SaveInterval = args['SaveInterval']
        self.LearningRate = args['LearningRate']
        self.WeightDecay = args['WeightDecay']
        self.BatchSize = args['BatchSize']
        self.EmaRate = args['EmaRate']
        self.DataDropRate = args['DataDropRate']
        self.DataDropDelay = args['DataDropDelay']
        self.LRDecay = args['LRDecay']
        self.Lambda = args['Lambda']
       
        self.Device = Device
        self.Model = Model
        self.Diffusor = Diffusor

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


        self.Optimizer = torch.optim.AdamW(Model.parameters(),lr=self.LearningRate,weight_decay=self.WeightDecay)
        self.Scheduler = torch.optim.lr_scheduler.ExponentialLR(self.Optimizer,self.LRDecay)
        self.Criterion = torch.nn.MSELoss()

        self.EmaModel = copy.deepcopy(Model).eval().requires_grad_(False)

    def TrainLoop(self,Dataset: DataLoader) -> None:

        Time1 = time.time()
        DataSetLength = len(Dataset)

        LossList = []
        Step = 0

        for Epoch in range(0,self.Epoch):

            for i,(Cond,OutImage) in enumerate(Dataset,0):

                if Step > self.DataDropDelay:
                    Mask = torch.rand([self.BatchSize]) > self.DataDropRate
                    Cond = Cond * Mask.int()[:,None,None,None]

                Cond = Cond.to(self.Device)
                OutImage = OutImage.to(self.Device)

                T = self.Diffusor.RandomTimeStep(self.BatchSize).to(self.Device)
                X,Noise = self.Diffusor.AddNoise(OutImage,T)
                
                Loss,Mse,Vlb = self.TrainStep(X,Noise,Cond,T,OutImage)
                LossList.append(Loss)

                if self.SessionPath != None:
                    self.Board.add_scalar("Hybrid", Loss, global_step=Step)
                    self.Board.add_scalar("MSE", Mse, global_step=Step)
                    self.Board.add_scalar("Vlb", Vlb, global_step=Step)
                    self.Logger.info("MSE:{} | VLB:{} | Loss:{} | Step:{} | Epoch:{} | Itteration:{}".format(Mse,Vlb,Loss,Step,Epoch,i))

                if i % self.LogInterval == 0:

                    AveLoss = np.mean(LossList)

                    Time2 = time.time()
                    Msg = 'Epoch: {} | Itteration: {} | Step: {} | average diffusion loss: {} | time(s):{} '.format(Epoch,i,Step,AveLoss,Time2-Time1)
                    print(Msg)
                    self.Logger.info(Msg)
                    Time1 = time.time()

                    LossList.clear()

                if Step % self.SaveInterval == 0 and self.SessionPath != None:
                    
                    self.Board.flush()
                    self.SaveModel(str(Step))
                    self.CreateTrainSamples(Dataset,Step)

                Step += 1

            self.Scheduler.step()
        
        if self.SavePath != None:
            self.SaveModel('Final')
            self.CreateTrainSamples(Dataset,Step)
            self.Board.close()

    def TrainStep(self,X,Y,Cond,T,X0) -> float:

        Output = self.Model(X,T,Cond)
        Mean,Var = torch.chunk(Output,2,dim=1)

        Mse = self.Criterion(Mean,Y)
        Vlb = self.Lambda * self.KLLoss(Var,Mean.detach(),T,X,X0)

        Loss = Mse + Vlb
        
        self.Optimizer.zero_grad()
        Loss.backward()
        self.Optimizer.step()
        
        #Exponential moving average model update
        for Weight,EmaWeight in zip(self.Model.parameters(),self.EmaModel.parameters()):
            EmaWeight.data = self.EmaRate * EmaWeight.data + (1.0-self.EmaRate) * Weight.data

        return Loss.item(), Mse.item() ,Vlb.item()
    
    def KLLoss(self,Varience,Mean,T,X,X0):

        TrueMean,TrueVar,TrueLogVar = self.Diffusor.postMeanVarience(X0,X,T)

        PredMean,PredVar,PredLogVar = self.Diffusor.SampleMeanVarience(T,X,Mean,Varience,Clip=False)
    
        Kl = self.KLLossNormal(TrueMean,TrueLogVar,PredMean,PredLogVar) 
        Kl = torch.mean(Kl,dim=[1,2,3]) / math.log(2.0)
        
        nll = -self.discretized_gaussian_log_likelihood(X0,means=PredMean,log_scales=0.5 * PredLogVar)
        nll = torch.mean(nll,dim=[1,2,3]) / math.log(2.0)

        Loss = torch.where((T==0),nll,Kl)

        Loss = Loss.mean()

        return Loss
    
    #-----------------
    # """
    # Helpers for various likelihood-based losses. These are ported from the original
    # Ho et al. diffusion models codebase:
    # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
    # """


    def KLLossNormal(self,TrueMean,TrueVar,PredMean,PredVar):
        return 0.5 * (-1.0 + PredVar - TrueVar + torch.exp(TrueVar - PredVar) + ((TrueMean - PredMean) ** 2) * torch.exp(-PredVar))

    def approx_standard_normal_cdf(self,x):
        return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x,3) )))

    def discretized_gaussian_log_likelihood(self,x, *, means, log_scales):
    
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs

    # -------------------

    
    def ReturnEmaModel(self):
        return self.EmaModel
    
    def CreateTrainSamples(self,Dataset:DataLoader,Step:int)->None:

        SampleCond = next(iter(Dataset))[0]
        ModelGrid = self.SampleLog(self.Model,self.BatchSize,SampleCond,"Model_{}".format(Step))
        EmaGrid = self.SampleLog(self.EmaModel,self.BatchSize,SampleCond,"EmaModel_{}".format(Step))

        self.Board.add_image("ModelTrainSamples",ModelGrid,dataformats='HWC',global_step=Step)
        self.Board.add_image("EmaModelTrainSamples",EmaGrid,dataformats='HWC',global_step=Step)
    
    def SampleLog(self,Model,N,Cond, Indicator:str):

        assert self.SessionPath != None

        self.Logger.info('Start Sampling Images for Logging')

        SampledImages = self.Diffusor.SampleImage(Model,N,Cond)

        SampleGrid = ImgTorchToNumpy(make_grid(SampledImages))

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


