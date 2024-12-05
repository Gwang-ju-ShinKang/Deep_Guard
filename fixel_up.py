
#Github git 클론해서 가져와 사용하기
!git clone https://github.com/cszn/BSRGAN.git

################

import torch
import cv2
import os
import sys
sys.path.append("/content/BSRGAN") # 이걸 지정하지 않으면 아래의 import가 기능하지 않음
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
import main_download_pretrained_models
###################

!python /content/BSRGAN/main_download_pretrained_models.py --models "BSRGAN" --model_dir "/content/BSRGAN/model_zoo"
!python /content/BSRGAN/main_download_pretrained_models.py --models "BSRGAN.pth" --model_dir "/content/BSRGAN/model_zoo"
###################

from google.colab import files
uploaded = files.upload()
####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load(os.path.join('/content/BSRGAN/model_zoo', 'BSRGAN.pth')), strict=True)
model = model.to(device)
model.eval()
#######################
for img_path in uploaded.keys():
  with torch.no_grad():
    img = cv2.imread(img_path)

    img_L = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_L = util.uint2tensor4(img_L)

    img_E = model(img_L)

    img_E = util.tensor2uint(img_E)
    util.imsave(img_E, os.path.splitext(img_path)[0] + '_result.png')
######################