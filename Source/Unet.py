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
        self.Conv1 = torch.nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.Encode1 = EncodeBlock(64,self.TimeHidden)

        self.Conv2 = torch.nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2)
        self.Encode2 = EncodeBlock(128,self.TimeHidden)

        self.Conv3= torch.nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2)
        self.Encode3 = EncodeBlock(256,self.TimeHidden)

        self.Conv4= torch.nn.Conv2d(256,512,kernel_size=3,padding=1,stride=2)
        self.Encode4 = EncodeBlock(512,self.TimeHidden)

        self.Conv5= torch.nn.Conv2d(512,512,kernel_size=3,padding=1,stride=2)
        self.Encode5 = EncodeBlock(512,self.TimeHidden)

        self.Conv6= torch.nn.Conv2d(512,512,kernel_size=3,padding=1,stride=2)
        self.Encode6 = EncodeBlock(512,self.TimeHidden)

        # #BOTTLENECK
        self.Decode1 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)
        self.Decode2 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)
        self.Decode3 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)

        # #DECODER
        self.ConvTran1 = torch.nn.ConvTranspose2d(512,512,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.Decode4 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)

        self.ConvTran2 = torch.nn.ConvTranspose2d(512,512,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.Decode5 = DecodeBlock(512,self.ConditionalClasses,self.TimeHidden)

        self.ConvTran3 = torch.nn.ConvTranspose2d(512,256,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.Decode6 = DecodeBlock(256,self.ConditionalClasses,self.TimeHidden)

        self.ConvTran4 = torch.nn.ConvTranspose2d(256,128,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.Decode7 = DecodeBlock(128,self.ConditionalClasses,self.TimeHidden)

        self.ConvTran5 = torch.nn.ConvTranspose2d(128,64,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.Decode8 = DecodeBlock(64,self.ConditionalClasses,self.TimeHidden)

        self.Decode9 = DecodeBlock(64,self.ConditionalClasses,self.TimeHidden)
        self.Conv9 = torch.nn.Conv2d(64,3,kernel_size=3,padding=1)


    def forward(self,x,T,Cond):

        EmbT = self.SinCosTime(T)

        # #ENCODER
        x = self.Conv1(x)
        x = self.Encode1(x,EmbT)
        skip1 = x

        x = self.Conv2(x)
        x = self.Encode2(x,EmbT)
        skip2 = x

        x = self.Conv3(x)
        x = self.Encode3(x,EmbT)
        skip3 = x

        x = self.Conv4(x)
        x = self.Encode4(x,EmbT)
        skip4 = x

        x = self.Conv5(x)
        x = self.Encode5(x,EmbT)
        skip5 = x

        x = self.Conv6(x)
        x = self.Encode6(x,EmbT)
        skip6 = x

        # #BOTTLENECK
        x = self.Decode1(x,Cond,EmbT)
        x = self.Decode2(x,Cond,EmbT)
        x = self.Decode3(x,Cond,EmbT)

        # #DECODE
        x = x + skip6
        x = self.ConvTran1(x)
        x = self.Decode4(x,Cond,EmbT)

        x = x + skip5
        x = self.ConvTran2(x)
        x = self.Decode5(x,Cond,EmbT)

        x = x + skip4
        x = self.ConvTran3(x)
        x = self.Decode6(x,Cond,EmbT)

        x = x + skip3
        x = self.ConvTran4(x)
        x = self.Decode7(x,Cond,EmbT)

        x = x + skip2
        x = self.ConvTran5(x)
        x = self.Decode8(x,Cond,EmbT)

        x = x + skip1
        x = self.Decode9(x,Cond,EmbT)
        x = self.Conv9(x)

        return x