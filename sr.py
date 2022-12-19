import torch 
import torchvision
from torchvision import transforms

import numpy as np 
import PIL
import os
import glob

# for super resolution
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchsr.models import ninasr_b0
from torchsr.models import edsr, rcan
from torchsr.models.utils import ChoppedModel, SelfEnsembleModel

"""
A python script that automatically scales the image by a factor of 4 using super-resolution technique
"""

# Specify the location of train or test folder
source_dir = 'test_shuffle'
out_dir = 'sr_test_shuffle/'
full_data_list = glob.glob(os.path.join(source_dir, '*.jpg'))

sr_model = SelfEnsembleModel(edsr(scale=4, pretrained=True))

count = 1
for f in full_data_list:
    print("Count " + str(count))
    img_id = f.split('/')[-1].split('.')[0]
    
    img = PIL.Image.open(f)
    img_np = np.asarray(img)
    # make tensor
    img_t = to_tensor(img_np).unsqueeze(0)
    # run SR
    img_sr = sr_model(img_t)
    # convert from tensor to PIL image
    img_ret = to_pil_image(img_sr.squeeze(0))
    img_ret.save(out_dir+img_id+".jpg")
    count += 1

    