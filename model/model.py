import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class BraTS2021BaseUnetModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[64, 128, 256, 512], focal_loss=False):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.focal_loss = focal_loss

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # for focal loss initialize bias of final_conv to b = − log((1 − a)/a), a = 0.01
        if self.focal_loss:
            print("Initializing final_conv bias for focal loss")
            a = torch.full_like(self.final_conv.bias, 0.01)
            self.final_conv.bias.data = -torch.log((1 - a) / a)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)

        # for focal loss we require log softmax
        if self.focal_loss:
            return F.log_softmax(x, dim=1)

        # our last layer is softmax, to get probabilities for each pixel for each class
        else:
            return F.softmax(x, dim=1)

## Attention U_net V1
class Attention_block(nn.Module):
    def __init__(self,g_ch,x_upper_layer,int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=g_ch, out_channels=int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=x_upper_layer, out_channels=int, kernel_size=1,stride=2,padding=0,bias=True),
            nn.BatchNorm2d(int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=int, out_channels=1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=x_upper_layer, out_channels=x_upper_layer, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(x_upper_layer)
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        psi = self.upsample(psi)
        out = self.last(x*psi)

        return out

class BraTS2021AttentionUnetModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[64, 128, 256, 512]):
        super().__init__()
        self.att = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):

            self.att.append(Attention_block(feature*2,feature,int=feature))

            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
           
            skip_connection = skip_connections[idx//2]

            att_output = self.att[idx//2](x, skip_connection)

            x = self.ups[idx](x) # reduce feature, increase image size. example: torch.Size([4, 1024, 15, 15]) -> torch.Size([4, 512, 30, 30])
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((att_output, x), dim=1) # example : after concat:torch.Size([4, 1024, 30, 30])
            x = self.ups[idx+1](concat_skip) #reduce feature,  image size keeps the same. example: torch.Size([4, 1024, 30, 30]) -> torch.Size([4, 512, 30, 30])

        x = self.final_conv(x)

        # our last layer is softmax, to get probabilities for each pixel for each class
        class_prob = F.softmax(x, dim=1)

        return class_prob

## Attention U_net V3
class Attention_block_V3(nn.Module):
    def __init__(self,g_ch,x_upper_layer,int):
        super(Attention_block_V3,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=g_ch, out_channels=int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=x_upper_layer, out_channels=int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=int, out_channels=1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=x_upper_layer, out_channels=x_upper_layer, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(x_upper_layer)
            )

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = self.last(x*psi)

        return out

class BraTS2021AttentionUnetModel_V3(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[64, 128, 256, 512]):
        super().__init__()
        self.att = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            
            self.att.append(Attention_block_V3(feature,feature,int=feature//2))

            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
           
            skip_connection = skip_connections[idx//2]
            # reduce feature, increase image size. example: torch.Size([4, 1024, 15, 15]) -> torch.Size([4, 512, 30, 30])
            x = self.ups[idx](x) 

            att_output = self.att[idx//2](x, skip_connection)
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((att_output, x), dim=1) # example : after concat:torch.Size([4, 1024, 30, 30])
            x = self.ups[idx+1](concat_skip) #reduce feature,  image size keeps the same. example: torch.Size([4, 1024, 30, 30]) -> torch.Size([4, 512, 30, 30])

        x = self.final_conv(x)

        # our last layer is softmax, to get probabilities for each pixel for each class
        class_prob = F.softmax(x, dim=1)

        return class_prob


## Attention U_net V4
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block_V4(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block_V4,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi,psi
        
class BraTS2021AttentionUnetModel_V4(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, plot_attention=False):
        super(BraTS2021AttentionUnetModel_V4,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block_V4(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block_V4(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block_V4(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block_V4(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.plot_attention = plot_attention


    def forward(self,x):
        # DOWN path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

         # # UP + concat path
        d5 = self.Up5(x5)
        x4,psi4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)

       
        d4 = self.Up4(x4)
        x3,psi3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2,psi2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,psi1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.transpose(d1,2,3)

        # our last layer is softmax, to get probabilities for each pixel for each class
        class_prob = F.softmax(d1, dim=1)

        if self.plot_attention:
            return class_prob, psi1, psi2, psi3, psi4
        else:
            return class_prob

def test():
    x = torch.randn((4, 4, 240, 240))
    model = BraTS2021AttentionUnetModel_V4()
    preds, psi1, psi2, psi3, psi4 = model(x)
    print ('preds :',preds.shape)
    print ('x :',x.shape)
    print ('psi1 :',psi1.shape)
    print ('psi2 :',psi2.shape)
    print ('psi3 :',psi3.shape)
    print ('psi4 :',psi4.shape)
    # x = torch.randn((4, 4, 240, 240))
    # g = torch.randn((4, 4, 240, 240))
    # preds : torch.Size([4, 4, 240, 240])
    # x : torch.Size([4, 4, 240, 240])
    # psi1 : torch.Size([4, 1, 240, 240])
    # psi2 : torch.Size([4, 1, 120, 120])
    # psi3 : torch.Size([4, 1, 60, 60])
    # psi4 : torch.Size([4, 1, 30, 30])


if __name__ == "__main__":
    test()