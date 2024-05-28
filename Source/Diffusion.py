import torch
import math

import numpy as np

from Source.Utils import TanReNormalize

class Diffusor():
    def __init__(self,args,Device='cpu') -> None:

        self.Timesteps = args['TimeSteps']
        self.ImageSize = args['ImageSize']
        self.NoiseSchedule = args['NoiseSchedule']

        self.Device = Device

        if self.NoiseSchedule == 'Linear': 
            self.Beta = self.LinearSchedule().to(self.Device)
        elif self.NoiseSchedule == 'Cosine':
            self.Beta = self.CosineSchedule().to(self.Device)
        else:
            raise NotImplementedError()
        
        self.Alpha = 1.0 - self.Beta
        self.AlphaHat = torch.cumprod(self.Alpha,dim=0)

        One = torch.tensor([1.0]).to(self.Device)
        self.AlphaHatPrev = torch.cat([One,self.AlphaHat])

    def RandomTimeStep(self,N):
        return torch.randint(0,self.Timesteps,size=[N,])
    
    def LinearSchedule(self,BetaStart = 1e-4,BetaEnd= 0.02):

        Scale = 1000 / self.Timesteps

        return torch.linspace(Scale * BetaStart,Scale * BetaEnd,self.Timesteps)
    
    def CosineSchedule(self,S = 0.008):
        steps = self.Timesteps+1
        X = torch.linspace(0,self.Timesteps,steps)
        AlphaHat = torch.cos((X/self.Timesteps)+S / (1+S)* math.pi * .5) ** 2
        AlphaHat = AlphaHat / AlphaHat[0]
        Beta = 1 - (AlphaHat[1:] / AlphaHat[:-1])
        return torch.clamp(Beta,0.0001,0.9999)

    def AddNoise(self,X,T):
        AlphaSquare = torch.sqrt(torch.gather(self.AlphaHat,0,index=T))[:,None,None,None]
        AlphaSquareMin = torch.sqrt(1.0 - torch.gather(self.AlphaHat,0,index=T))[:,None,None,None]
        e = torch.randn_like(X)

        return AlphaSquare * X + AlphaSquareMin * e, e
    
    def postMeanVarience(self,X0,X,T):

        Alpha = torch.gather(self.Alpha,0,T)[:,None,None,None]
        Alphahat = torch.gather(self.AlphaHat,0,T)[:,None,None,None]
        Alphahatprev = torch.gather(self.AlphaHatPrev,0,T)[:,None,None,None]
        Beta = torch.gather(self.Beta,0,T)[:,None,None,None]

        coef1 = Beta * torch.sqrt(Alphahatprev) / (1.0-Alphahat)
        coef2 = (torch.sqrt(Alpha) * (1 - Alphahatprev)) / (1.0-Alphahat)

        mean = coef1 * X0 + coef2* X

        var = Beta * (1.0 - Alphahatprev) / (1.0 - Alphahat)
        logvar = torch.log(torch.maximum(var,torch.tensor(1e-20)))

        return mean,var,logvar
    
    def SampleMeanVarience(self,T,X,Mean,Var,Clip=True):

        Alphahat = torch.gather(self.AlphaHat,0,T)[:,None,None,None]
        Beta = torch.gather(self.Beta,0,T)[:,None,None,None]

        X0 = torch.sqrt(1.0/Alphahat) * X - torch.sqrt(1.0 / Alphahat -1.0) * Mean

        if Clip == True:
            X0 = X0.clamp(-1,1)
            Var = Var.clamp(-1,1)

        clipmean,_,logpostvar = self.postMeanVarience(X0,X,T)

        Var = TanReNormalize(Var)
        logvariance = torch.log(Beta) * Var + (1-Var) * logpostvar
        variance = torch.exp(logvariance)

        return clipmean,variance,logvariance
    
    def SampleImage(self,Model :torch.nn.Module,N: int, Cond, CFGscale = 3.0,ReturnSteps = False,ImageSize = None):

        Model.eval()

        Cond = Cond.to(self.Device)

        Steps = []

        if ImageSize == None:
            StartSize = self.ImageSize
        else:
            StartSize = ImageSize 

        with torch.no_grad():
            X = torch.randn((N,3,StartSize[1],StartSize[0])).to(self.Device)
            for i in reversed(range(0,self.Timesteps)):

                T = (torch.ones(N)*i).long().to(self.Device)

                Out = Model(X,T,Cond)
                Mean , Var = torch.chunk(Out,2,dim=1)

                if CFGscale > 1.0:
                    CFGOut = Model(X,T,torch.zeros_like(Cond))
                    CFGMean,CFGVar = torch.chunk(CFGOut,2,dim=1)

                    Mean = torch.lerp(CFGMean,Mean,CFGscale)

                if i == 0:
                    Noise = torch.zeros_like(X)
                else:
                    Noise = torch.randn_like(X)

                clipmean,_,logvariance = self.SampleMeanVarience(T,X,Mean,Var)

                X = clipmean + torch.exp(logvariance * 0.5) * Noise

                if ReturnSteps:
                    Step = TanReNormalize(X.clamp(-1,1)).to('cpu')
                    Steps.append(Step)

        X = TanReNormalize(X.clamp(-1,1)).to('cpu')

        Model.train()

        if ReturnSteps == False:
            return X
        else:
            return X , Steps