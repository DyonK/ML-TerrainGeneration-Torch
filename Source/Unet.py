import torch

from Source.Modules import *

class TimeEmbSequential(torch.nn.Sequential):

    def forward(self,x,Cond,T):

        for Layer in self:
            if isinstance(Layer,DecodeBlock):
                x = Layer(x,Cond,T)
            elif isinstance(Layer,EncodeBlock):
                x = Layer(x,T)
            else:
                x = Layer(x)
        return x

class Unet(torch.nn.Module):
    def __init__(self,args) -> None:
        super(Unet , self).__init__()

        self.TimeDim = args['TimeDims']
        self.TimeHidden = self.TimeDim * 4
        self.ConditionalClasses = args['ConditionalClasses']

        self.SinCosTime = TimeEmbedding(self.TimeDim,self.TimeHidden)

        # #ENCODER
        self.EncodeBlock1 = TimeEmbSequential(
                torch.nn.Conv2d(3,16,kernel_size=3,padding=1),
                EncodeBlock(16,self.TimeHidden,16),
                EncodeBlock(16,self.TimeHidden,16)
            )

        self.EncodeBlock2 = TimeEmbSequential(
                torch.nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2),
                EncodeBlock(32,self.TimeHidden),
                EncodeBlock(32,self.TimeHidden)
            )

        self.EncodeBlock3 = TimeEmbSequential(
                torch.nn.Conv2d(32,64,kernel_size=3,padding=1,stride=2),
                EncodeBlock(64,self.TimeHidden),
                EncodeBlock(64,self.TimeHidden)
            )
        
        self.EncodeBlock4 = TimeEmbSequential(
                torch.nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2),
                EncodeBlock(128,self.TimeHidden),
                EncodeBlock(128,self.TimeHidden)
            )
        
        self.EncodeBlock5 = TimeEmbSequential(
                torch.nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2),
                EncodeBlock(256,self.TimeHidden),
                EncodeBlock(256,self.TimeHidden),
                QKVAttention(256)
            )
        
        self.EncodeBlock6 = TimeEmbSequential(
                torch.nn.Conv2d(256,512,kernel_size=3,padding=1,stride=2),
                EncodeBlock(512,self.TimeHidden),
                EncodeBlock(512,self.TimeHidden),
                QKVAttention(512)
            )

        # #BOTTLENECK
       
        self.MiddleBlock = TimeEmbSequential(
                DecodeBlock(512,self.ConditionalClasses,self.TimeHidden),
                QKVAttention(512),
                DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)
            )

        # #DECODER

        self.DecodeBlock1 = TimeEmbSequential(
            torch.nn.ConvTranspose2d(512,256,kernel_size=3,padding=1,stride=2,output_padding=1),
            DecodeBlock(256,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(256,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(256,self.ConditionalClasses,self.TimeHidden),
            QKVAttention(256)
        )

        self.DecodeBlock2 = TimeEmbSequential(
            torch.nn.ConvTranspose2d(256,128,kernel_size=3,padding=1,stride=2,output_padding=1),
            DecodeBlock(128,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(128,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(128,self.ConditionalClasses,self.TimeHidden),
            QKVAttention(128)
        )

        self.DecodeBlock3 = TimeEmbSequential(
            torch.nn.ConvTranspose2d(128,64,kernel_size=3,padding=1,stride=2,output_padding=1),
            DecodeBlock(64,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(64,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(64,self.ConditionalClasses,self.TimeHidden)
        )

        self.DecodeBlock4 = TimeEmbSequential(
            torch.nn.ConvTranspose2d(64,32,kernel_size=3,padding=1,stride=2,output_padding=1),
            DecodeBlock(32,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(32,self.ConditionalClasses,self.TimeHidden),
            DecodeBlock(32,self.ConditionalClasses,self.TimeHidden)
        )

        self.DecodeBlock5 = TimeEmbSequential(
            torch.nn.ConvTranspose2d(32,16,kernel_size=3,padding=1,stride=2,output_padding=1),
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16),
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16),
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16)
        )

        self.DecodeBlock6 = TimeEmbSequential(
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16),
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16),
            DecodeBlock(16,self.ConditionalClasses,self.TimeHidden,16),
            torch.nn.Conv2d(16,3,kernel_size=3,padding=1)
        )



    def forward(self,x,T,Cond):

        EmbT = self.SinCosTime(T)

        # #ENCODER
      
        Enc1 = self.EncodeBlock1(x,Cond,EmbT)
        Enc2 = self.EncodeBlock2(Enc1,Cond,EmbT)
        Enc3 = self.EncodeBlock3(Enc2,Cond,EmbT)
        Enc4 = self.EncodeBlock4(Enc3,Cond,EmbT)
        Enc5 = self.EncodeBlock5(Enc4,Cond,EmbT)
        Enc6 = self.EncodeBlock6(Enc5,Cond,EmbT)

        # #BOTTLENECK

        Mid = self.MiddleBlock(Enc6,Cond,EmbT)

        # #DECODE

        Dec1 = self.DecodeBlock1(Mid + Enc6,Cond,EmbT)
        Dec2 = self.DecodeBlock2(Dec1 + Enc5,Cond,EmbT)
        Dec3 = self.DecodeBlock3(Dec2 + Enc4,Cond,EmbT)
        Dec4 = self.DecodeBlock4(Dec3 + Enc3,Cond,EmbT)
        Dec5 = self.DecodeBlock5(Dec4 + Enc2,Cond,EmbT)
        Dec6 = self.DecodeBlock6(Dec5 + Enc1,Cond,EmbT)
        

        return Dec6