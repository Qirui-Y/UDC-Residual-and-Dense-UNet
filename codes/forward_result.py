import torch
from models import create_model
import options.options as option
import numpy as np
import utils.util as util

visualization = False
if visualization:
    import matplotlib.pyplot as plt


opt = dict()
opt['model'] = 'sr'
opt['is_train'] = False
opt['dist'] = False
opt['train'] = {}

opt['gpu_ids'] = 1  # change here
opt['network_G'] = {'which_model_G':'UDC_res_5'}  # change here
opt['path'] = {'pretrain_model_G':'model/Poled_model.pth','strict_load':True}  # change here
model = create_model(opt)

def get_result(img_LQ):
    # input: [H, W, 3], RGB
    img_input = img_LQ.copy()
    img_LQ = img_LQ/255
    img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float() # [3, H, W]
    data = {}
    data['LQ'] = img_LQ.unsqueeze(0)   # [1, 3, H, W]
    
    model.feed_data(data)
    model.test()
    visuals = model.get_current_visuals()
    #sr_img = util.tensor2img(visuals['SR'])  # uint8
    img_np = visuals['SR'].numpy()
    img_np = np.transpose(img_np[:, :, :], (1, 2, 0))  # HWC, RGB
    img_np = (img_np * 255.0).round()
    sr_img = img_np.astype(np.uint8)
    
    if visualization:
        plt.subplot(1,2,1)
        plt.imshow(img_input)
        plt.subplot(1,2,2)
        plt.imshow(sr_img)
        plt.show()
    
    return sr_img
    