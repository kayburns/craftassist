'''
tester.py

Test the trained 3dgan models
'''

import torch
from torch import optim
from torch import nn
from collections import OrderedDict
import os
import gc
import glob
from model import *
from datasets import Craft3DDataset, Craft3DDatasetAnno

# added
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import params
import visdom
import faiss
from tqdm import tqdm
import pickle as pkl

################################################################################
# Utilities
################################################################################

def load_pretrained_GE(save_file_path, G, E):

    pretrained_file_path_G = save_file_path+'/'+'G.pth'
    pretrained_file_path_E = save_file_path+'/'+'E.pth'
    
    if not torch.cuda.is_available():
        G.load_state_dict(
            torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}),
            strict=False)
        E.load_state_dict(
            torch.load(pretrained_file_path_E, map_location={'cuda:0': 'cpu'}),
            strict=False)
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        E.load_state_dict(torch.load(pretrained_file_path_E))
 
    return G, E

def occupancy_accuracy(gt, pred):
    """Compute occupancy accuracy of (batched) torch tensors."""
    return (1 - torch.nn.L1Loss()(gt, pred))

def occupancy_iou(gt, pred):
    """Compute occupancy iou of (batched) torch tensors."""
    intersection = ((gt == pred) & (gt > 0)).sum((-1,-2,-3)).float()
    union = gt.sum((-1,-2,-3)) + pred.sum((-1,-2,-3)) - intersection
    union[union == 0] = 1
    intersection[union == 0] = 1
    return torch.mean((intersection / union))

################################################################################
# Test Methods
################################################################################

def tester_generate_osl_bank(args):

    if args.debug:
        import pdb; pdb.set_trace()
        num_workers = 0
    else:
        num_workers = 1

    image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    save_file_path = params.output_dir + '/' + args.model_name
   
    G = net_G_blocks(args)
    E = net_E(args)
    load_pretrained_GE(save_file_path, G, E)
    #E.to(params.device)
    #G.to(params.device)
    G.eval()
    E.eval()


    train_dsets = Craft3DDatasetAnno(params.data_dir, "train", remove="roof") 
    train_dset_loader1 = torch.utils.data.DataLoader(
        train_dsets, batch_size=1, shuffle=False, num_workers=num_workers)

    n = 1
    zref_diff_avg = torch.zeros(200).float()

    for i, (ref_pre, ref_post) in enumerate(tqdm(train_dset_loader1)):

        #ref_pre.to(params.device)
        #ref_post.to(params.device)
        zref_pre = E(ref_pre)
        zref_post = E(ref_post)

        zref_diff = zref_pre - zref_post #  this is addition, counterintuitively
        zref_diff_avg += (zref_diff - zref_diff_avg) / n
        n += 1
    zref_diff_avg = zref_diff_avg.data.numpy()
    t_dict = {'roof': zref_diff_avg}
    pkl.dump(
        t_dict,
        open("/craftassist/python/threedhouses/data/gen_bank/gen_bank.pkl", "wb"))

            
