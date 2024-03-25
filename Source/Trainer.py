import torch
import time
import copy

from torch.utils.data import DataLoader
from Source.Diffusion import Diffusor

from Source.Utils import ShowMap,TanReNormalize,TanNormalize

import numpy as np

class Trainer():
    def __init__(self,args,Model : torch.nn.Module, Diffusor : Diffusor) -> None:

        self.Epoch = args.TrainEpoch
        self.LogInterval = args.LogInterval
        self.SaveInterval = args.SaveInterval
        self.LearningRate = args.LearningRate
        self.WeightDecay = args.WeightDecay
        self.BatchSize = args.BatchSize
        self.EmaRate = args.EmaRate
        self.Device = args.Device

        self.Model = Model
        self.Diffusor = Diffusor

        self.Optimizer = torch.optim.AdamW(Model.parameters(),lr=self.LearningRate,weight_decay=self.WeightDecay)
        self.Criterion = torch.nn.MSELoss()

        self.EmaModel = copy.deepcopy(Model).eval().requires_grad_(False)

    def TrainLoop(self,Dataset: DataLoader) -> None:

        for Epoch in range(0,self.Epoch):

            Time1 = time.time()
            EpochLosses = []

            for i,(Cond,OutImage) in enumerate(Dataset,0):

                Cond = Cond.to(self.Device)
                OutImage = OutImage.to(self.Device)

                T = self.Diffusor.RandomTimeStep(self.BatchSize).to(self.Device)
                X,Noise = self.Diffusor.AddNoise(OutImage,T)
                
                Loss = self.TrainStep(X,Noise,Cond,T)
                EpochLosses.append(Loss)


                if i % self.LogInterval == 0:

                    AveLoss = np.mean(EpochLosses)

                    Time2 = time.time()
                    print('Epoch: {} | Itteration: {} | average diffusion loss: {} | time(s):{} '.format(Epoch,i,AveLoss,Time2-Time1))
                    Time1 = time.time()

                    EpochLosses.clear()

    def TrainStep(self,X,Y,Cond,T) -> float:

        self.Optimizer.zero_grad()

        Output = self.Model(X,T,Cond)

        Loss = self.Criterion(Output,Y)
        Loss.backward()

        self.Optimizer.step()

        #Exponential moving average model update
        for Weight,EmaWeight in zip(self.Model.parameters(),self.EmaModel.parameters()):
            EmaWeight.data = self.EmaRate * EmaWeight.data + (1.0-self.EmaRate) * Weight.data


        return Loss.item()
    
    def ReturnEmaModel(self):
        return self.EmaModel