# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import gc
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
import cv2

from models.detection_models import networks
#from models.detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")

class DetectionModel():
    def __init__(self, root="tmp" , input_path = "input", output_path = "mask",  checkpoint_name = 'unet.pt', gpu = 0):
        self.in_dir = os.path.join(root, input_path)
        self.out_dir = os.path.join(root, output_path)
        self.checkpoint_name = checkpoint_name
        self.device = "gpu" if gpu>=0 else "cpu"
        self.gpu = 0

    def inference(self, filename=None):
        print("initializing the dataloader")
        # set up model
        self.model = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )

        # load model
        checkpoint_path = os.path.join("checkpoints", self.checkpoint_name)
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        print("model weights loaded")

        # device 
        if self.device=="gpu":
            self.model.to(self.gpu)
        else: 
            self.model.cpu()
        
        self.model.eval()
        
        if (filename==None):    
            self.batchProc()
        else:
            self.pieceProc(filename)

    def batchProc(self):
        imagelist = os.listdir(self.in_dir)
        imagelist.sort()

        idx = 0
        for image_name in imagelist:
            idx += 1
            print("processing", image_name)
            self.pieceProc(image_name)
        del self.model

    def pieceProc(self, filename):
        # Load Image
        scratch_file = os.path.join(self.in_dir, filename)
        
        if not os.path.isfile(scratch_file):
            print("Skipping non-file %s" % filename)
            return 
        scratch_image = Image.open(scratch_file).convert("RGB")
        w, h = scratch_image.size

        # PIL -> Tensor
        # transformed_image_PIL = data_transforms(scratch_image, config.input_size)
        scratch_image = scratch_image.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = scale_tensor(scratch_image)

        # device
        if self.gpu >= 0:
            scratch_image_scale = scratch_image_scale.to(self.gpu)
        else:
            scratch_image_scale = scratch_image_scale.cpu()
        
        # inference
        with torch.no_grad():
            P = torch.sigmoid(self.model(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")
        self.save((P >= 0.4).int()[0][0], filename)
        
        gc.collect()
        torch.cuda.empty_cache()

    def save(self, mask, filename):
        cv2.imwrite(os.path.join(self.out_dir, filename), mask.numpy())



if __name__ == "__main__":
    dmHelper = DetectionModel()
    dmHelper.inference("real_1.png")
