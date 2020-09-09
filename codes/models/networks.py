import torch
import logging
import models.modules.UDC_densePix as UDC_densePix
import models.modules.UDC_ResPix as UDC_ResPix

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
                                    
    if which_model == 'UDC_dense_3':
        netG = UDC_densePix.UDC_dense_3()

    elif which_model == 'UDC_dense_4':
        netG = UDC_densePix.UDC_dense_4()

    elif which_model == 'UDC_dense_5':
        netG = UDC_densePix.UDC_dense_5()

    elif which_model == 'UDC_res_3':
        netG = UDC_ResPix.UDC_res_3()

    elif which_model == 'UDC_res_4':
        netG = UDC_ResPix.UDC_res_4()

    elif which_model == 'UDC_res_5':
        netG = UDC_ResPix.UDC_res_5()
                                  
    
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
