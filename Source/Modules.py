import torch
import torchvision





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
