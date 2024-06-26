import torch

from Source.Modules import *

class Unet(torch.nn.Module):
    def __init__(self,args) -> None:
        super(Unet , self).__init__()

        self.TimeDim = args['TimeDims']
        self.TimeHidden = self.TimeDim * 4
        self.ConditionalClasses = args['ConditionalClasses']
        self.Channels = args['Channels']
        self.Blocks = args['ResBlocks']
        self.AttentionThresh = args['AttentionThreshhold']
        
        self.SinCosTime = TimeEmbedding(self.TimeDim,self.TimeHidden)

        ##ENCODER
        self.EncoderList = torch.nn.ModuleList()
        self.EncoderList.append(torch.nn.Conv2d(3,self.Channels[0],kernel_size=3,padding=1))

        PrevChannels = self.Channels[0]
        SkipChannels = [self.Channels[0]]
        
        for i,Channels in enumerate(self.Channels):

            for j in range(0,self.Blocks):

                Att = i > self.AttentionThresh
                self.EncoderList.append(EncodeBlock(PrevChannels,Channels,self.TimeHidden,Attention=Att))

                PrevChannels = Channels
                SkipChannels.append(Channels)

            if i != len(self.Channels)-1:
                
                self.EncoderList.append(torch.nn.Conv2d(Channels,Channels,kernel_size=3,stride=2,padding=1))
                SkipChannels.append(Channels)

            PrevChannels = Channels

        ## BOTTLENECK

        BottleAtt = len(self.Channels) > self.AttentionThresh + 1
        self.BottleList = torch.nn.ModuleList()
        for i in range(0,self.Blocks):
            self.BottleList.append(DecodeBlock(self.Channels[-1],self.Channels[-1],self.ConditionalClasses,self.TimeHidden,Attention=BottleAtt))
        
        # #DECODER
        self.DecoderList = torch.nn.ModuleList()
        DecChannels = list(reversed(self.Channels))

        PrevChannels = DecChannels[0]

        for i,Channels in enumerate(DecChannels):

            for j in range(0,self.Blocks+1):

                InChannels = PrevChannels + SkipChannels.pop()
                Att = i < len(DecChannels) - (self.AttentionThresh +1)
                self.DecoderList.append(DecodeBlock(InChannels,Channels,self.ConditionalClasses,self.TimeHidden,Attention=Att))

                PrevChannels = Channels

            if i != len(DecChannels)-1:
                self.DecoderList.append(torch.nn.ConvTranspose2d(Channels,Channels,kernel_size=3,stride=2,padding=1,output_padding=1))

            PrevChannels = Channels
        
        self.Out = torch.nn.Sequential(
            torch.nn.GroupNorm(32,PrevChannels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(PrevChannels,6,3,padding=1)
        )

    def forward(self,x,T,Cond):

        Cond = Cond.float()
        EmbT = self.SinCosTime(T)

        # #ENCODER 
        Skips = []
        for Layer in self.EncoderList:
            if isinstance(Layer,EncodeBlock):
                x = Layer(x,EmbT)
            else:
                x = Layer(x)
            
            Skips.append(x)
        
        #BOTTLENECK
        for Layer in self.BottleList:
            x = Layer(x,Cond,EmbT)

        #DECODE
        for Layer in self.DecoderList:

            if isinstance(Layer,DecodeBlock):

                x = torch.concat([x , Skips.pop()],dim=1)

                x = Layer(x,Cond,EmbT)
            else:
                x = Layer(x)

        Out = self.Out(x)

        return Out




