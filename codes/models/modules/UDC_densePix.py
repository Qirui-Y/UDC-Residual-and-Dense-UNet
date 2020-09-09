import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = mutil.make_layer(basic_block, nb)


        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first, self.HRconv, self.conv_last], 0.1)
        

    def forward(self, x):
        #print('x:',x.size())
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        out = self.conv_last(self.lrelu(self.HRconv(out)))

        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        #mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5  + x


class UDC_dense_3(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(UDC_dense_3, self).__init__()
        
        # encoder
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(ResidualDenseBlock_5C, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 4)                    # 64f   
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(ResidualDenseBlock_5C, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 4)                   # 128f
       
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(ResidualDenseBlock_5C, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 4)                   # 256f
        
        self.conv_4 = nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True)    # 256f
        #self.ca_4 = CALayer(channel=256, reduction=1)

        self.conv_5 = nn.Conv2d(nf*4, 256, 3, 1, 1, bias=True)       # 256f
        
        # pooling
        self.avg_pool = nn.AvgPool2d(2)   

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)      
        
        # decoder
        self.conv_6 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        #self.ca_5 = CALayer(channel=256, reduction=1)

        self.conv_7 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        #self.ca_6 = CALayer(channel=128, reduction=1)

        self.conv_9 = nn.Conv2d(384, 256, 3, 1, padding=1, bias=True) ##
        self.conv_10 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        #self.ca_7 = CALayer(channel=64, reduction=1)
       
        self.conv_11 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=True) ##
        self.conv_12 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        #self.ca_8 = CALayer(channel=32, reduction=1)

        self.pixshuffle_1 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_2 = nn.PixelShuffle(upscale_factor=2)
  
        self.pixshuffle_3 = nn.PixelShuffle(upscale_factor=2)

        self.conv_13 = nn.Conv2d(96, 64, 3, 1, padding=1, bias=True)
        self.conv_14 = nn.Conv2d(64, 3, 3, 1, padding=1, bias=True)
  

    def forward(self, x):
        B, C, H, W = x.size()
        x_in = x
        #x_mean = torch.mean(x, dim=[2,3])   # [B, 3]
        #x_std = torch.std(x, dim=[2,3])   # [B, 3]
        #x_mean = x_mean.unsqueeze(2).unsqueeze(2)  # [B, 3, 1, 1]
        #x_std = x_std.unsqueeze(2).unsqueeze(2)    # [B, 3, 1, 1]
        
        # encoder
        fea = self.lrelu(self.conv_1(x))
        fea_cat1 = self.encoder1(fea)                       # [B, 64, H, W]
       
        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea)) 
        fea_cat2 = self.encoder2(fea)                       # [B, 128, H/2, W/2]
       
        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))
        fea_cat3 = self.encoder3(fea)                       # [B, 256, H/16, W/16]
        
        fea = self.avg_pool(fea_cat3)
        fea_cat4 = self.conv_4(fea)        # [B, 256, H/8, W/8]
        fea = self.lrelu(fea_cat4)
        fea = self.conv_5(fea)             # [B, 256, H/8, W/8]
        
        # decoder
        de_fea = (self.conv_6(fea))                         # [B, 256, H/8, W/8]
       
        de_fea_cat1 = torch.cat([fea_cat4, de_fea], 1)      # [B, 512, H/8, W/8]

        de_fea = self.lrelu((self.conv_7(de_fea_cat1)))     # [B, 512, H/8, W/8]
        de_fea = (self.conv_8(de_fea))                      # [B, 512, H/8, W/8]
        de_fea = self.lrelu(self.pixshuffle_1(de_fea))      # [B, 128, H/8, W/8]
        de_fea_cat2 = torch.cat([fea_cat3, de_fea], 1)      # [B, 384, H/4, W/4] 

        de_fea = self.lrelu((self.conv_9(de_fea_cat2)))     # [B, 256, H/4, W/4]    
        de_fea = (self.conv_10(de_fea))                     # [B, 256, H/4, W/4]
        de_fea = self.lrelu(self.pixshuffle_2(de_fea))      # [B, 64, H/4, W/4]  
        de_fea_cat3 = torch.cat([fea_cat2, de_fea], 1)      # [B, 192, H/2, W/2] 

        de_fea = self.lrelu((self.conv_11(de_fea_cat3)))    # [B, 128, H/2, W/2]
        de_fea = (self.conv_12(de_fea))                     # [B, 128, H/2, W/2]
        de_fea = self.lrelu(self.pixshuffle_3(de_fea))      # [B, 32, H/2, W/2]
        de_fea_cat4 = torch.cat([fea_cat1, de_fea], 1)      # [B, 96, H, W]

        de_fea = self.lrelu((self.conv_13(de_fea_cat4)))    # [B, 64, H, W]
        out = self.conv_14(de_fea)                     # [B, 3, H, W]
        #print('de_fea: ', de_fea.size())
        #print(out.shape)
                
        return out

class UDC_dense_4(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(UDC_dense_4, self).__init__()
        
        # encoder
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(ResidualDenseBlock_5C, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(ResidualDenseBlock_5C, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(ResidualDenseBlock_5C, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f

        self.conv_4 = nn.Conv2d(nf*4, nf*8, 3, 1, 1, bias=True)                # 512f
        basic_block_512 = functools.partial(ResidualDenseBlock_5C, nf=nf*8) 
        self.encoder4 = mutil.make_layer(basic_block_512, 2)                   # 512f
        
        self.conv_5 = nn.Conv2d(nf*8, nf*8, 3, 1, 1, bias=True)    # 64f
        self.conv_6 = nn.Conv2d(nf*8, 512, 3, 1, 1, bias=True)       # 64f
        
        # pooling
        self.avg_pool = nn.AvgPool2d(2)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # decoder
        self.conv_7 = nn.Conv2d(512, 256, 3, 1, padding=1, bias=True)
        
        self.conv_8 = nn.Conv2d(768, 512, 3, 1, padding=1, bias=True)

        self.conv_9 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        
        self.conv_10 = nn.Conv2d(640, 512, 3, 1, padding=1, bias=True)

        self.conv_11 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
       
        self.conv_12 = nn.Conv2d(384, 256, 3, 1, padding=1, bias=True)

        self.conv_13 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        
        self.conv_14 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=True)

        self.conv_15 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
       
        self.conv_16 = nn.Conv2d(96, 64, 3, 1, padding=1, bias=True)
       
        self.conv_17 = nn.Conv2d(64, 3, 3, 1, padding=1, bias=True)


        self.pixshuffle_1 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_2 = nn.PixelShuffle(upscale_factor=2)
       
        self.pixshuffle_3 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_4 = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        B, C, H, W = x.size()
        x_in = x
        #x_mean = torch.mean(x, dim=[2,3])   # [B, 3]
        #x_std = torch.std(x, dim=[2,3])   # [B, 3]
        #x_mean = x_mean.unsqueeze(2).unsqueeze(2)  # [B, 3, 1, 1]
        #x_std = x_std.unsqueeze(2).unsqueeze(2)    # [B, 3, 1, 1]
        # encoder
        fea = self.lrelu(self.conv_1(x))    #[B, 64, H, W]
        fea_cat1 = self.encoder1(fea)      # [B, 64, H, W]

        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea))   
        fea_cat2 = self.encoder2(fea)      # [B, 128, H/2, W/2]

        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))   #
        fea_cat3 = self.encoder3(fea)      # [B, 256, H/4, W/4]

        fea = self.avg_pool(fea_cat3)
        fea = self.lrelu(self.conv_4(fea))   #
        fea_cat4 = self.encoder4(fea)      # [B, 512, H/8, W/8]

        fea = self.avg_pool(fea_cat4)
        fea_cat5 = self.conv_5(fea)        # [B, 256, H/16, W/16]
       
        fea = self.lrelu(fea_cat5)
        fea = self.conv_6(fea)             # [B, 256, H/8, W/8]
        
        # decoder
        de_fea = (self.conv_7(fea))                       # [B, 256, H/16, W/16]
        de_fea_cat1 = torch.cat([fea_cat5, de_fea], 1)    # [B, 512, H/16, W/16]
        de_fea = self.lrelu((self.conv_8(de_fea_cat1)))   # [B, 512, H/16, W/16]
        de_fea = (self.conv_9(de_fea))                    # [B, 512, H/16, W/16]
        de_fea = self.lrelu(self.pixshuffle_1(de_fea))    # [B, 128, H/16, W/16]

        de_fea_cat2 = torch.cat([fea_cat4, de_fea], 1)    # [B, 640, H/8, W/8]       
        de_fea = self.lrelu((self.conv_10(de_fea_cat2)))   # [B, 512, H/8, W/8] 
        de_fea = (self.conv_11(de_fea))                    # [B, 512, H/8, W/8]                   
        de_fea = self.lrelu(self.pixshuffle_2(de_fea))    # [B, 128, H/8, W/8]

        de_fea_cat3 = torch.cat([fea_cat3, de_fea], 1)    # [B, 384, H/4, W/4]       
        de_fea = self.lrelu((self.conv_12(de_fea_cat3)))  # [B, 256, H/4, W/4]   
        de_fea = (self.conv_13(de_fea))                    # [B, 256, H/4, W/4] 
        de_fea = self.lrelu(self.pixshuffle_3(de_fea))    # [B, 64, H/4, W/4]  

        de_fea_cat4 = torch.cat([fea_cat2, de_fea], 1)      # [B, 192, H/2, W/2]          
        de_fea = self.lrelu((self.conv_14(de_fea_cat4)))    # [B, 128, H/2, W/2]
        de_fea = (self.conv_15(de_fea))                      # [B, 128, H/2, W/2]
        de_fea = self.lrelu(self.pixshuffle_4(de_fea))      # [B, 32, H/2, W/2]
        

        de_fea_cat5 = torch.cat([fea_cat1, de_fea], 1)    # [B, 96, H, W]
        de_fea = self.lrelu((self.conv_16(de_fea_cat5)))    # [B, 64, H, W]
        out = self.conv_17(de_fea)                     # [B, 3, H, W]
        #print('de_fea: ', de_fea.size())               
        return out

class UDC_dense_5(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(UDC_dense_5, self).__init__()
        #CA
        modules_body_1 = []
        modules_body_1.append(CALayer(nf, reduction=8))
        self.body_1 = nn.Sequential(*modules_body_1)

        # encoder
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(ResidualDenseBlock_5C, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(ResidualDenseBlock_5C, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(ResidualDenseBlock_5C, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f

        self.conv_4 = nn.Conv2d(nf*4, nf*8, 3, 1, 1, bias=True)                # 512f
        basic_block_512 = functools.partial(ResidualDenseBlock_5C, nf=nf*8) 
        self.encoder4 = mutil.make_layer(basic_block_512, 2)                   # 512f

        self.conv_5 = nn.Conv2d(nf*8, nf*16, 3, 1, 1, bias=True)                # 1024f
        basic_block_1024 = functools.partial(ResidualDenseBlock_5C, nf=nf*16) 
        self.encoder5 = mutil.make_layer(basic_block_1024, 2)     # 1024f
        
        self.conv_6 = nn.Conv2d(nf*16, nf*8, 3, 1, 1, bias=True)    # 512f
        self.conv_7 = nn.Conv2d(nf*8, nf*8, 3, 1, 1, bias=True)       # 256f
        
        # pooling
        self.avg_pool = nn.AvgPool2d(2)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder
        self.conv_8 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        
        self.conv_9 = nn.Conv2d(1024, 768, 3, 1, padding=1, bias=True)
        
        self.conv_10 = nn.Conv2d(768, 512, 3, 1, padding=1, bias=True)
       
        self.conv_11 = nn.Conv2d(1152,768, 3, 1, padding=1, bias=True)
        
        self.conv_12 = nn.Conv2d(768, 512, 3, 1, padding=1, bias=True)
       
        self.conv_13 = nn.Conv2d(640, 256, 3, 1, padding=1, bias=True)
       
        self.conv_14 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        
        self.conv_15 = nn.Conv2d(320, 256, 3, 1, padding=1, bias=True)
      
        self.conv_16 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)

        self.conv_17 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=True)
      
        self.conv_18 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)

        self.conv_19 = nn.Conv2d(96, 64, 3, 1, padding=1, bias=True)

        self.conv_20= nn.Conv2d(64, 3, 3, 1, padding=1, bias=True)
        
        self.pixshuffle_1 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_2 = nn.PixelShuffle(upscale_factor=2)
       
        self.pixshuffle_3 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_4 = nn.PixelShuffle(upscale_factor=2)
        
        self.pixshuffle_5 = nn.PixelShuffle(upscale_factor=2)
       
        

    def forward(self, x):
        B, C, H, W = x.size()
        x_in = x
    
        # encoder
        fea = self.lrelu(self.conv_1(x))    #[B, 64, H, W]
        fea_cat1 = self.encoder1(fea)      # [B, 64, H, W]

        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea))   
        fea_cat2 = self.encoder2(fea)      # [B, 128, H/2, W/2]

        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))   #
        fea_cat3 = self.encoder3(fea)      # [B, 256, H/4, W/4]

        fea = self.avg_pool(fea_cat3)
        fea = self.lrelu(self.conv_4(fea))   #
        fea_cat4 = self.encoder4(fea)      # [B, 512, H/8, W/8]

        fea = self.avg_pool(fea_cat4)
        fea = self.lrelu(self.conv_5(fea))   #
        fea_cat5 = self.encoder5(fea)      # [B, 1024, H/16, W/16]

        fea = self.avg_pool(fea_cat5)
        fea_cat6 = self.conv_6(fea)        # [B,512, H/32, W/32]
       
        fea = self.lrelu(fea_cat6)
        fea = self.conv_7(fea)             # [B, 256, H/32, W/32]       
        
        # decoder
        de_fea = (self.conv_8(fea))       # [B, 256, H/32, W/32]   
        de_fea_cat1 = torch.cat([fea_cat6, de_fea], 1)    # [B, 768, H/32, W/32]          
        de_fea = self.lrelu((self.conv_9(de_fea_cat1)))    # [B, 64, H/32, W/32]
        de_fea = (self.conv_10(de_fea))                     # [B, 64, H/32, W/32]
        de_fea = self.lrelu(self.pixshuffle_1(de_fea))
        #de_fea = F.upsample(de_fea, size=(H//16, W//16), mode='bilinear')

        de_fea_cat2 = torch.cat([fea_cat5, de_fea], 1)    # [B, 1088, H/16, W/16]
        de_fea = self.lrelu((self.conv_11(de_fea_cat2)))     # [B, 256, H/16, W/16]
        de_fea = (self.conv_12(de_fea))                      # [B, 64, H/16, W/16]
        de_fea = self.lrelu(self.pixshuffle_2(de_fea))
        #de_fea = F.upsample(de_fea, size=(H//8, W//8), mode='bilinear')

        de_fea_cat3 = torch.cat([fea_cat4, de_fea], 1)    # [B, 576, H/8, W/8]       
        de_fea = self.lrelu((self.conv_13(de_fea_cat3)))     # [B, 256, H/8, W/8]    
        de_fea = (self.conv_14(de_fea))                     # [B, 64, H/8, W/8]
        de_fea = self.lrelu(self.pixshuffle_3(de_fea))
        #de_fea = F.upsample(de_fea, size=(H//4, W//4), mode='bilinear')

        de_fea_cat4 = torch.cat([fea_cat3, de_fea], 1)    # [B, 320, H/4, W/4]       
        de_fea = self.lrelu((self.conv_15(de_fea_cat4)))     # [B, 128, H/4, W/4]    
        de_fea = (self.conv_16(de_fea))                     # [B, 64, H/4, W/4]
        de_fea = self.lrelu(self.pixshuffle_4(de_fea))
        #de_fea = F.upsample(de_fea, size=(H//2, W//2), mode='bilinear')

        de_fea_cat5 = torch.cat([fea_cat2, de_fea], 1)    # [B, 192, H/2, W/2]          
        de_fea = self.lrelu((self.conv_17(de_fea_cat5)))    # [B, 64, H/2, W/2]
        de_fea = (self.conv_18(de_fea))                     # [B, 64, H/2, W/2]
        de_fea = self.lrelu(self.pixshuffle_5(de_fea))
        #de_fea = F.upsample(de_fea, size=(H, W), mode='bilinear')

        de_fea_cat6 = torch.cat([fea_cat1, de_fea], 1)    # [B, 128, H, W]     
        de_fea = self.lrelu((self.conv_19(de_fea_cat6)))    # [B, 64, H, W]
        out = self.conv_20(de_fea)                     # [B, 3, H, W]
        #print('de_fea: ', de_fea.size())
    
        return out
