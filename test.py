import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import cv2 
from torch.utils import data
from collections import OrderedDict

from fcn import fcn8s

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict



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
	    
    n_classes = 7 

    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img).float()
    # Setup Model
    model = fcn8s(n_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)

    # Save results
    maps = ['vessel','od','ma','he','ex','se']

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    i=0
    for item in maps:
        i=i+1
        prob = outputs.data.cpu().numpy()[0,i,:,:]
        if args.dataset=='drive':
            misc.imsave(args.out_folder + "/result_" + item + '.png',(prob[:,:565]*255).astype(np.uint8))
        else:
            misc.imsave(args.out_folder + "/result_" + item + '.png',(prob*255).astype(np.uint8))



    print("Segmentation Masks Saved at: {}".format(args.out_folder))



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
        "--out_folder",
        nargs="?",
        type=str,
        default=None,
        help="Folder for the output segmap",
    )
    args = parser.parse_args()
    test(args)