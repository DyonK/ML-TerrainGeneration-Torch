import torch

import numpy as np

from typing import Optional
from Source.Utils import TanReNormalize

class Diffusor():
    def __init__(self,args) -> None:

        self.Timesteps = args.TimeSteps
        self.ImageSize = args.ImageSize

        self.Device = args.Device

        if args.NoiseSchedule == 'Linear': #TODO: cosine schedule
            self.Beta = self.LinearSchedule().to(self.Device)
        else:
            raise NotImplementedError()
        
        self.Alpha = 1.0 - self.Beta
        self.AlphaHat = torch.cumprod(self.Alpha,dim=0)

        One = torch.tensor([1.0]).to(self.Device)
        self.AlphaHatPrev = torch.cat([One,self.AlphaHat])
        

    def RandomTimeStep(self,N):
        return torch.randint(0,self.Timesteps,size=[N,])
    
    def LinearSchedule(self,BetaStart = 1e-4,BetaEnd= 0.02):
        return torch.linspace(BetaStart,BetaEnd,self.Timesteps)

    def AddNoise(self,X,T):
        AlphaSquare = torch.sqrt(torch.gather(self.AlphaHat,0,index=T))[:,None,None,None]
        AlphaSquareMin = torch.sqrt(1.0 - torch.gather(self.AlphaHat,0,index=T))[:,None,None,None]
        e = torch.randn_like(X)

        return AlphaSquare * X + AlphaSquareMin * e, e
    
    def SampleImage(self,Model :torch.nn.Module,N: int, Cond, CFGscale = 2.0):

        Model.eval()

        Cond = Cond.to(self.Device)

        with torch.no_grad():
            X = torch.randn((N,3,self.ImageSize[1],self.ImageSize[0])).to(self.Device)
            for i in reversed(range(0,self.Timesteps)):

                T = (torch.ones(N)*i).long().to(self.Device)

                PredNoise = Model(X,T,Cond)

                if CFGscale > 0.0:
                    PredNoiseCFG = Model(X,T,torch.zeros_like(Cond))
                    PredNoise = torch.lerp(PredNoiseCFG,PredNoise,CFGscale)

                Alpha = torch.gather(self.Alpha,0,T)[:,None,None,None]
                Alphahat = torch.gather(self.AlphaHat,0,T)[:,None,None,None]
                Alphahatprev = torch.gather(self.AlphaHatPrev,0,T)[:,None,None,None]
                Beta = torch.gather(self.Beta,0,T)[:,None,None,None]

                if i == 0:
                    Noise = torch.zeros_like(X)
                else:
                    Noise = torch.randn_like(X)

                X0 = torch.sqrt(1.0/Alphahat) * X - torch.sqrt(1.0 / Alphahat -1.0) * PredNoise
                X0 = X0.clamp(-1,1)

                coef1 = Beta * torch.sqrt(Alphahatprev) / (1.0-Alphahat)
                coef2 = (torch.sqrt(Alpha) * (1 - Alphahatprev)) / (1.0-Alphahat)

                mean = coef1 * X0 + coef2* X

                variance = Beta * (1.0 - Alphahatprev) / (1.0 - Alphahat)
                logvariance = torch.log(torch.maximum(variance,torch.tensor(1e-20))) 
                X = mean + torch.exp(logvariance * 0.5) * Noise
        
        X.clamp(-1,1)
        X = TanReNormalize(X)
        Model.train()

        return X
        
        

                
  