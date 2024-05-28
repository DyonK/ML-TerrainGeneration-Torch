import torch

from Source.Modules import *

class Unet(torch.nn.Module):
    def __init__(self,args) -> None:
        super(Unet , self).__init__()

        self.TimeDim = args['TimeDims']
        self.TimeHidden = self.TimeDim * 4
        self.ConditionalClasses = args['ConditionalClasses']

        self.SinCosTime = TimeEmbedding(self.TimeDim,self.TimeHidden)

        # #ENCODER  
        self.In = torch.nn.Conv2d(3,64,kernel_size=3,padding=1)
        
        self.Enc1 = EncodeBlock(64,self.TimeHidden)
        self.Down1= torch.nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2)
        
        self.Enc2 = EncodeBlock(128,self.TimeHidden)
        self.Down2 = torch.nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2)
        
        self.Enc3 = EncodeBlock(256,self.TimeHidden)
        self.Down3 = torch.nn.Conv2d(256,512,kernel_size=3,padding=1,stride=2)
        
        self.Enc4 = EncodeBlock(512,self.TimeHidden)

        # #DECODER
        self.Dec1 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)
        self.Dec2 = DecodeBlock(512+512,self.ConditionalClasses,self.TimeHidden)
        self.Up1 =  torch.nn.ConvTranspose2d(512+512,256,kernel_size=3,padding=1,stride=2,output_padding=1)
        
        self.Dec3 = DecodeBlock(256+256,self.ConditionalClasses,self.TimeHidden)
        self.Dec4 = DecodeBlock(512+256,self.ConditionalClasses,self.TimeHidden)
        self.Up2 = torch.nn.ConvTranspose2d(512+256,128,kernel_size=3,padding=1,stride=2,output_padding=1)
        
        self.Dec5 = DecodeBlock(128+128,self.ConditionalClasses,self.TimeHidden)
        self.Dec6 = DecodeBlock(256+128,self.ConditionalClasses,self.TimeHidden)
        self.Up3 = torch.nn.ConvTranspose2d(256+128,64,kernel_size=3,padding=1,stride=2,output_padding=1)

        self.Dec7 = DecodeBlock(64+64,self.ConditionalClasses,self.TimeHidden)
        self.Dec8 = DecodeBlock(128+64,self.ConditionalClasses,self.TimeHidden)

        self.Out = torch.nn.Sequential(
            torch.nn.GroupNorm(32,128+64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128+64,6,3,padding=1)
        )

    def forward(self,x,T,Cond):

        EmbT = self.SinCosTime(T)

        In = self.In(x)

        # #ENCODER
        Enc1 = self.Enc1(In,EmbT)
        Down1 = self.Down1(Enc1)

        Enc2 = self.Enc2(Down1,EmbT)
        Down2 = self.Down2(Enc2)

        Enc3 = self.Enc3(Down2,EmbT)
        Down3 = self.Down3(Enc3)

        Enc4 = self.Enc4(Down3,EmbT)
        
        # #DECODE
        Dec1 = self.Dec1(Enc4,Cond,EmbT)
        Dec2 = self.Dec2(torch.cat([Dec1,Down3],dim=1),Cond,EmbT)
        Up1 = self.Up1(Dec2)

        Dec3 = self.Dec3(torch.cat([Up1,Enc3],dim=1),Cond,EmbT)
        Dec4 = self.Dec4(torch.cat([Dec3,Down2],dim=1),Cond,EmbT)
        Up2 = self.Up2(Dec4)

        Dec5 = self.Dec5(torch.cat([Up2,Enc2],dim=1),Cond,EmbT)
        Dec6 = self.Dec6(torch.cat([Dec5,Down1],dim=1),Cond,EmbT)
        Up3 = self.Up3(Dec6)

        Dec7 = self.Dec7(torch.cat([Up3,Enc1],dim=1),Cond,EmbT)
        Dec8 = self.Dec8(torch.cat([Dec7,In],dim=1),Cond,EmbT)

        Out = self.Out(Dec8)

        return Out




