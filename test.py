import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2 
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )

import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    imag = misc.imread(args.img_path)
    if args.dataset=='drive':
        img = np.zeros((584,880,3))
        img[:,:565,:]=imag
    elif args.dataset=='idrid':
        img=imag
    else:
        img = misc.imresize(imag, (584,880))
	    
    data_loader = get_loader('drive')
    data_path = '../DRIVE'
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = 7 
    resized_img = img.astype(np.uint8)

    orig_size = img.shape[:-1]

    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img).float()
    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)
    print(np.unique(outputs.data.cpu().numpy()))

    
    prob = outputs.data.cpu().numpy()[0,1,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob.png",(prob*255).astype(np.uint8))

    prob = outputs.data.cpu().numpy()[0,6,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob5.png",(prob*255).astype(np.uint8))

    prob = outputs.data.cpu().numpy()[0,2,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob1.png",(prob*255).astype(np.uint8))

    prob = outputs.data.cpu().numpy()[0,3,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob2.png",(prob*255).astype(np.uint8))

    prob = outputs.data.cpu().numpy()[0,4,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob3.png",(prob*255).astype(np.uint8))

    prob = outputs.data.cpu().numpy()[0,5,:,:]
    print(np.unique(prob*255))
    misc.imsave(args.out_path[:-4] + "_prob4.png",(prob*255).astype(np.uint8))

    print("Segmentation Mask Saved at: {}".format(args.out_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="drive",
        help="Dataset to use",
    )
    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)
