import torch

from Source.Modules import Spade

class Unet(torch.nn.Module):
    def __init__(self,args) -> None:
        super(Unet , self).__init__()

        self.TimeDim = args.TimeDims

        self.Conv1 = torch.nn.Conv2d(3,64,3,padding=1)
        self.Relu1 = torch.nn.ReLU()

        self.Spade1 = Spade(64,args.ConditionalClasses)

        self.Conv2 = torch.nn.Conv2d(64,3,3,padding=1)
        




    def forward(self,x,T,Cond):

        x = self.Conv1(x)
        x = self.Relu1(x)

        x = self.Spade1(x,Cond)

        x = self.Conv2(x)

        return x