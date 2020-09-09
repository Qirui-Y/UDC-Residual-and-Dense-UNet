#!/usr/bin/env python
#Code Partially Forked from NTIRE2020 Denoising Challenge @ CVPR
#Revised by Yuqian Zhou
import numpy as np
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat
import time

import forward_result


total_time = 0
sum_runtime = 0
avg_runtime = 0

def restoration(udc):
    # TODO: plug in your method here
    # udc [H, W, 3]
    start = time.time()
    result = forward_result.get_result(udc)
    end = time.time()
    
    during = end - start
    global total_time
    total_time = total_time + during
    
    runtime = during/(udc.shape[0]*udc.shape[1]/1000000)
    global sum_runtime
    sum_runtime = sum_runtime + runtime
    print('forward one time: %.4f  run time: %.4f seconds/megapixel'%(during,runtime))
    
    return result


# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = './'

# load noisy images
udc_fn = 'poled_test_display.mat'  # or poled_val_display.mat
udc_key = 'test_display'
udc_mat = loadmat(os.path.join(work_dir, udc_fn))[udc_key]

# restoration
n_im, h, w, c = udc_mat.shape
results = udc_mat.copy()
for i in range(n_im):
    udc = np.reshape(udc_mat[i, :, :, :], (h, w, c))
    
    restored = restoration(udc)
    results[i, :, :, :] = restored

avg_runtime = sum_runtime/n_im
print('Total time: %.4f  Total images: %d   FPS: %.4f   Run time: %.4f seconds/megapixel'%(total_time,n_im,total_time/n_im, avg_runtime))

# create results directory
res_dir = 'results/res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission indormation
# TODO: update the values below; the evaluation code will parse them
runtime = 0  # seconds / megapixel
cpu_or_gpu = 0  # 0: GPU, 1: CPU
method = 1  # 0: traditional methods, 1: deep learning method
other = '(optional) any additional description or information'

# prepare and save readme file
readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
with open(readme_fn, 'w') as readme_file:
    readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
    readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
    readme_file.write('Method: %s\n' % str(method))
    readme_file.write('Other description: %s\n' % str(other))

# compress results directory
res_zip_fn = 'results_dir'
shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))