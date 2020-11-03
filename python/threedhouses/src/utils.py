'''
utils.py

Some utility functions

'''

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import params

if params.device.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import csv
import pickle
import matplotlib.cm as cm

def getVoxelFromMat(path, cube_len=64):
    if cube_len == 64:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        # voxels = np.load(path) 
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
    # print (voxels.shape)
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', marker='s', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

def visualize_schematic(voxels, path, iteration, labels=None, btype=False):
    voxels = voxels[:8]
    if not btype:
        voxels = voxels.__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)
    color_grid = cm.rainbow(np.arange(255))

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        if labels:
            ax = plt.subplot(gs[i], projection='3d', title=labels[i])
        else:
            ax = plt.subplot(gs[i], projection='3d')
        if btype:
            bl = sample[x, y, z].astype(int)
            color = color_grid[bl]
            ax.scatter(y, z, x, zdir='z', marker='s', c=color)
        else:
            ax.scatter(y, z, x, zdir='z', marker='s', c='r')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

def visualize_schematic_recon(X, recon, path, iteration):
    recon = recon[:4]#.__ge__(0.5)
    X = X[:4]
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)
    color_grid = cm.rainbow(np.arange(255))

    voxels = np.concatenate((X, recon), axis=0)
    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        b = sample[x, y, z].astype(int)
        #print(b.max())
        color = color_grid[b]

        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(y, z, x, zdir='z', marker='s', c=color)
        #ax.scatter(y, z, x, zdir='z', marker='s')
        #ax.set_xlim(0, 64)
        #ax.set_ylim(0, 64)
        #ax.set_zlim(0, 64)
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/recon_{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

def visualize_schematic_latent_noise(X, X_hats, path, iteration):
    X_hats = [X_hat.__ge__(0.5) for X_hat in X_hats]
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)
    color_grid = cm.rainbow(np.arange(255))

    X_hats = np.concatenate(X_hats, axis=0)
    voxels = np.concatenate((X, X_hats), axis=0)
    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        #b = sample[x, y, z].astype(int)
        #print(b.max())
        #color = color_grid[b]

        ax = plt.subplot(gs[i], projection='3d')
        #ax.scatter(y, z, x, zdir='z', marker='s', c=color)
        ax.scatter(y, z, x, zdir='z', marker='s', c='red')
        #ax.set_xlim(0, 64)
        #ax.set_ylim(0, 64)
        #ax.set_zlim(0, 64)
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/embs_{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


def SaveRecon_Voxels(X, recon, path, iteration):
    recon = recon[:4].__ge__(0.5)
    X = X[:4]
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)
    voxels = np.concatenate((X, recon), axis=0)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(y, z, x, zdir='z', marker='s', c='red')
        #ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

class ShapeNetDataset(data.Dataset):

    def __init__(self, root, args, train_or_val="train"):
        
        
        self.root = root
        self.listdir = os.listdir(self.root)
        # print (self.listdir)  
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
#        self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) # train: 10668-1000=9668
        self.args = args

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, params.cube_len), dtype=np.float32)
            # print (volume.shape)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)

################################################################################
########                          Dataset                             ##########
################################################################################

def minimal_dataset_loader():
    data_dir = '/scr/kayburns/instance_segmentation_data/'
    split_fname = 'training_data.pkl'
    splits_path = osp.join(data_dir, split_fname)

    with open(splits_path, "rb") as f:
        houses = pkl.load(f)
    
    return houses

def create_block_type_remap():
    blockname2id = create_blockname2id()
    block_remap_names = {}
    with open("block_type_map.txt", "r") as f:
        for line in f:
            new_name, old_names = line.split("\"")[1], line.split("\"")[3]
            block_remap_names[new_name] = old_names.split(",")
    block_remap = {}
    for new_name, old_names in block_remap_names.items():
        old_ids = [blockname2id[n] for n in old_names]
        block_remap[blockname2id[new_name]] = old_ids
    return block_remap
    
def read_blockid2clsid(fpath):
    blockid2clsid = {}
    with open(fpath, "r") as f:
        reader = csv.DictReader(csvfile)
        for row in reader:
            blockid2clsid[int(row['bid'])] = int(row['clsid'])
    return blockid2clsid

def create_blockname2id():
    blockname2id = {}
    with open('blocks.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['id2']) == 0: # only use canonical category
                blockname2id[row['name']] = int(row['id1'])
    return blockname2id

def read_counts():
    bname2count = {}
    with open("block_type_map.txt", "r") as f:
        for line in f:
            bname, count = line.split(",")
            bname2count[bname] = int(count)
    return bname2count

def write_dict(dic, fname):
    with open(fname, 'w+') as f:
        for k,v in dic.items():
            f.write(str(k) + ',' + str(v) + '\n')

def create_bid2clsid_500f_post_remap():
    classes = ["air","iron trapdoor","oak stairs","stone","oak planks","torch","bed","grass","glass","chest","fence"]
    clsname2clsid = dict(zip(classes, range(len(classes))))
    blockname2id = create_blockname2id()
    id2blockname = dict((reversed(b2id) for b2id in blockname2id.items()))
    postremap_id2preremap_id = create_block_type_remap()
    preremap_id2postremap_id = {}
    for new_idx, old_idxs in postremap_id2preremap_id.items():
        for old_idx in old_idxs:
            preremap_id2postremap_id[old_idx] = new_idx

    bid2clsid = {}
    for bid in id2blockname.keys():
        if bid in preremap_id2postremap_id.keys():
            remapped_bid = preremap_id2postremap_id[bid]
            bname = id2blockname[remapped_bid]
        else:
            bname = id2blockname[bid]
        if bname in classes:
            bid2clsid[bid] = classes.index(bname)
        else:
            bid2clsid[bid] = len(classes)

    write_dict(bid2clsid, 'bid2clsid_500f_post_remap.txt')

################################################################################
########                          VAE                                 ##########
################################################################################

def generateZ(args, batch):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z

def sample_gaussian(m, v):
    zeros, ones = torch.zeros_like(m), torch.ones_like(v)
    return torch.sqrt(v) * torch.normal(zeros, ones) + m

class KLNormal(nn.Module):

    def __init__(self):
        super(KLNormal, self).__init__()

    def forward(self, qm, qv, pm, pv):
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

