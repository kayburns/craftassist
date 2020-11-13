import torch
from torch import optim
from torch import nn
from collections import OrderedDict
import os
import gc
import glob
import pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threedhouses.src.model import *
from threedhouses.src.datasets import Craft3DDatasetAnno

# added
import datetime
import numpy as np
import threedhouses.src.params
from tqdm import tqdm
import pickle as pkl

##################################### Utils ####################################

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

def plot_house(pth, house):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*house.nonzero())
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    plt.savefig(pth)
    plt.clf()
 

class TransformationBank():

    def __init__(self, pth, load=False):

        self.save_path = os.path.join(pth, 'gen_bank.pkl')
        self.t_dict = {}
        if load:
            self.load_tranformation_bank(self.save_path)

    def load_tranformation_bank(self, pth):
        
        if os.path.exists(pth):
            self.t_dict = pkl.load(open(pth, 'rb'))

    def get_proposal(self, label):

        if label in self.t_dict:
            return self.t_dict[label]
        else:
            return None

    def contains(self, ref_obj):
        return ref_obj in self.t_dict

    def update(self, label, change_vector):
        self.t_dict[label] = change_vector


class GeneratorWrapper():

    def __init__(self):

        save_file_path = '/craftassist/python/threedhouses/models/ae_debug/'
        self.G, self.E = net_G_blocks([]), net_E([]) # args not used in model
        load_pretrained_GE(save_file_path, self.G, self.E)
        self.transformation_bank = TransformationBank(
            '/craftassist/python/threedhouses/data/gen_bank/')
        #self.transformation_bank = Craft3DDatasetAnno(
        #    params.data_dir, "train", remove="roof")

    def seen(self, ref_obj):
        """Return true if ref_obj in memory store"""
        return self.transformation_bank.contains(ref_obj)

    def update(self, label, blocks, house):

        # convert to 32 square cube
        offsets = self.find_block_offsets(house)
        X_pre = self.convert_blocks_to_arr(house, offsets)
        X_post = self.convert_blocks_to_arr(house + blocks, offsets)

        change_vector = self.fetch_change_vector(X_pre, X_post)
        self.transformation_bank.update(label, change_vector)

    def fetch_change_vector(self, X_pre, X_post):
        X_pre = torch.from_numpy(X_pre).float()
        X_post = torch.from_numpy(X_post).float()
        zref_pre, zref_post = self.E(X_pre), self.E(X_post)

        return zref_post - zref_pre

    def generate_build_proposal(self, X_pre):

        ref_pre, ref_post = self.transformation_bank[0]

        X_pre = torch.from_numpy(X_pre).float()
        zref_pre, zref_post = self.E(ref_pre), self.E(ref_post)
        zref_diff = zref_post - zref_pre

        zX_pre = self.E(X_pre)
        zXhat = zX_pre + zref_diff
        Xhat = self.G(zXhat)[0]

        # compute diff and threshld predictions
        Xhat[X_pre != 0] = 0
        Xhat[Xhat < .8] = 0

        return Xhat.data.numpy()

    def find_block_offsets(self, blocks):
        # fetch all coordinates and block types from node
        coords, blocks = zip(*blocks)
        blocks = [b[0] for b in blocks] # ignore meta block info
        coords = np.array(coords)

        # snap to corner
        min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
        xlen, zlen, ylen = max_coords - min_coords
        offsets = min_coords

        # standardize. format is (x, z, y). only standardize along (x, y)
        x_std, y_std = max(0, 16 - (xlen//2)), max(0, 16 - (ylen//2))
        offsets = offsets - np.array([x_std, 0, y_std])

        return offsets

    def convert_blocks_to_arr(self, blocks, offsets):
        coords, blocks = zip(*blocks)
        blocks = [b[0] for b in blocks] # ignore meta block info
        coords = np.array(coords)
        coords = coords - offsets
        coords = tuple(coords.T)

        cube_len = 32
        X = np.zeros([cube_len] * 3)
        X[coords] = blocks
        X = np.transpose(X, (1,2,0))

        return X
        
    def convert_arr_to_blocks(self, arr, shift):
        arr = np.transpose(arr,(2, 0, 1))
        min_coords, max_coords, x_std, y_std = shift
        coords = np.array(np.nonzero(arr)).T
        coords = coords - [x_std, 0, y_std]
        coords = coords + min_coords
        return coords

    def generate_build_proposal_wrapper(self, ref_block_dict):
        X_pre, shift = self.convert_blocks_to_arr(ref_block_dict)
        X_post = self.generate_build_proposal(X_pre)
        new_blocks = self.convert_arr_to_blocks(X_post, shift)
        return new_blocks.tolist()

