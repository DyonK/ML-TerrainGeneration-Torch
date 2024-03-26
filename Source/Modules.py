import torch
import torchvision
import math

from typing import Optional

class EncodeBlock(torch.nn.Module):
    def __init__(self,Channels,TimeChannels) -> None:
        super(EncodeBlock,self).__init__()

        self.Channels = Channels

        self.Norm1 = torch.nn.GroupNorm(32,self.Channels)

        self.Silu1 = torch.nn.SiLU()
        self.Conv1 = torch.nn.Conv2d(self.Channels,self.Channels,kernel_size=3,padding=1)

        self.Norm2 = torch.nn.GroupNorm(32,self.Channels)
        self.TimeEmb1 = TimeScaleShift(TimeChannels,self.Channels)

        self.Silu2 = torch.nn.SiLU()
        self.Conv2 = torch.nn.Conv2d(self.Channels,self.Channels,kernel_size=3,padding=1)

    def forward(self,x,T):

        skip = x

        x = self.Norm1(x)

        x = self.Silu1(x)
        x = self.Conv1(x)

        x = self.Norm2(x)
        x = self.TimeEmb1(x,T)

        x = self.Silu2(x)
        x = self.Conv2(x)

        x = x + skip

        return x

class DecodeBlock(torch.nn.Module):
    def __init__(self,Channels,CondChannels,TimeChannels) -> None:
        super(DecodeBlock,self).__init__()

        self.Channels = Channels

        self.Spade1 = Spade(self.Channels,CondChannels)

        self.Silu1 = torch.nn.SiLU()
        self.Conv1 = torch.nn.Conv2d(self.Channels,self.Channels,kernel_size=3,padding=1)

        self.Spade2 = Spade(self.Channels,CondChannels)
        self.TimeEmb1 = TimeScaleShift(TimeChannels,self.Channels)

        self.Silu2 = torch.nn.SiLU()
        self.Conv2 = torch.nn.Conv2d(self.Channels,self.Channels,kernel_size=3,padding=1)

    def forward(self,x,Cond,T):

        skip = x

        x = self.Spade1(x,Cond)

        x = self.Silu1(x)
        x = self.Conv1(x)

        x = self.Spade2(x,Cond)
        x = self.TimeEmb1(x,T)

        x = self.Silu2(x)
        x = self.Conv2(x)

        x = x + skip

        return x

class TimeScaleShift(torch.nn.Module):
    def __init__(self,InChannels,OutChannels) -> None:
        super(TimeScaleShift,self).__init__()

        self.InChannels = InChannels
        self.OutChannels = OutChannels * 2

        self.Seq1 = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.InChannels,self.OutChannels)
        )

    def forward(self,x,T):

        T = self.Seq1(T)[:,:,None,None]

        Scale,Shift = torch.chunk(T,2,dim=1)

        x = (x * (Scale + 1.0) + Shift)

        return x

class TimeEmbedding(torch.nn.Module):
    def  __init__(self,TimeDims,LearnedEmbDims : Optional[int]) -> None:
        super(TimeEmbedding,self).__init__()

        self.TimeDims = TimeDims

        if LearnedEmbDims != None:
            self.Seq1 = torch.nn.Sequential(
                torch.nn.Linear(TimeDims,LearnedEmbDims),
                torch.nn.SiLU(),
                torch.nn.Linear(LearnedEmbDims,LearnedEmbDims)
            )
            self.HiddenEmb = True
        else:
            self.HiddenEmb = False

    def forward(self,x):

        x = self.TimeStepEmbedding(x,self.TimeDims)

        if self.HiddenEmb == True:
            x = self.Seq1(x) 

        return x

    def TimeStepEmbedding(self,TimeSteps,Dimentions,MaxPeroid=10000):

        Halfchannels = Dimentions // 2 

        Exponent = -math.log(MaxPeroid) * torch.arange(start=0,end=Halfchannels,dtype=torch.float32).to(device=TimeSteps.device)

        Exponent = Exponent / Halfchannels
        Frequencies = torch.exp(Exponent)

        Embedding = TimeSteps[:,None].float() * Frequencies[None,:]

        Embedding = torch.cat([torch.sin(Embedding),torch.cos(Embedding)],dim=-1)

        return Embedding

class Spade(torch.nn.Module):
    def __init__(self,InChannels,CondChannels,HiddenChannels=128) -> None:
        super(Spade,self).__init__()

        self.GroupNorm1 = torch.nn.GroupNorm(32,InChannels,affine=False)

        self.Conv1 = torch.nn.Conv2d(CondChannels,HiddenChannels,kernel_size=3,padding=1)
        self.Relu1 = torch.nn.ReLU()

        self.ConvG = torch.nn.Conv2d(HiddenChannels,InChannels,kernel_size=3,padding=1)
        self.ConvB = torch.nn.Conv2d(HiddenChannels,InChannels,kernel_size=3,padding=1)

    def forward(self,x,Cond):

        x = self.GroupNorm1(x)

        NewSize = x.shape[2:]
        Cond = torchvision.transforms.Resize(NewSize,torchvision.transforms.InterpolationMode.NEAREST)(Cond).float()

        sx = self.Conv1(Cond)
        sx = self.Relu1(sx)

        Gamma = self.ConvG(sx)
        Beta = self.ConvB(sx)

        return (x * (Gamma + 1.0) + Beta)
    

