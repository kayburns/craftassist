'''
tester.py

Test the trained 3dgan models
'''

import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import *
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

# def test_gen(args):
#     test_z = []
#     test_num = 1000
#     for i in range(test_num):
#         z = generateZ(args, 1)
#         z = z.numpy()
#         test_z.append(z)
    
#     test_z = np.array(test_z)
#     print (test_z.shape)
    # np.save("test_z", test_z)

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

def tester_one_shot(args, eval_func):

    if args.debug:
        import pdb; pdb.set_trace()
        num_workers = 0
    else:
        num_workers = 1

    image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    G = net_G(args)
    E = net_E(args)
    G.eval()
    E.eval()

    save_file_path = params.output_dir + '/' + args.model_name
    load_pretrained_GE(save_file_path, G, E)

    train_dsets = Craft3DDatasetAnno(params.data_dir, "train") 
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=4, shuffle=False, num_workers=num_workers)

    for i, X in enumerate(tqdm(train_dset_loaders)):
        
        zx = E(X)
        zy = zx + delta
        Y_hats = G(zy)
        Y_hats = Y_hat_s.detach().unsqueeze(0).cpu().numpy()

        ### visualization
        voxels = torch.cat((X, Y_hats), dim=0)
        visualize_schematic(voxels, image_saved_path, i)

    
    # pixel accuracy for segmentation

    # pixel accuracy for house

    # noise

    return None

def tester(args):
    print ('Evaluation Mode...')

    # image_saved_path = '../images'
    image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom == True:
        vis = visdom.Visdom()

    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path+'/'+'G.pth'
    pretrained_file_path_D = save_file_path+'/'+'D.pth'
    
    print (pretrained_file_path_G)

    D = net_D(args)
    G = net_G(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    
    print ('visualizing model')
    
    # test generator
    # test_gen(args)
    G.to(params.device)
    D.to(params.device)
    G.eval()
    D.eval()

    # test_z = np.load("test_z.npy")
    # print (test_z.shape)
    # N = test_z.shape[0]

    N = 8

    for i in range(N):
        # z = test_z[i,:]
        # z = torch.FloatTensor(z)
        
        z = generateZ(args, 1)
        
        # print (z.size())
        fake = G(z)
        samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
        # print (samples.shape)
        # print (fake)
        y_prob = D(fake)
        y_real = torch.ones_like(y_prob)
        # criterion = nn.BCELoss()
        # print (y_prob.item(), criterion(y_prob, y_real).item())

        ### visualization
        if not args.use_visdom:
            visualize_schematic(samples, image_saved_path, 'tester_norm_'+str(i))

            # format schematic
            samples = samples.__ge__(0.3)
            samples = np.squeeze(samples, 0)

            x, y, z = samples.shape
            num_blocks = len(samples.nonzero()[0])
            block_array = np.concatenate(
                (np.ones((num_blocks, 1)), np.zeros((num_blocks, 1))),
                axis=1
            )

            schematic = np.zeros((x, y, z, 2))
            schematic[samples.nonzero()] = block_array
            schem_dict = {'name':['patio'],'tags':['patio'], 'schematic':schematic}

            schem_save_path = os.path.join(params.images_dir, "patio_%d.pkl" % i)
            with open(schem_save_path, 'wb') as f:
                pkl.dump(schem_dict, f)
        else:
            plotVoxelVisdom(samples[0,:], vis, "tester_"+str(i))
 
def save_embeddings(args):

    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_E = save_file_path+'/'+'E.pth'
    E = net_E(args)

    embeddings_file_path = save_file_path + "/embeddings/"
    if not os.path.exists(embeddings_file_path):
        os.makedirs(embeddings_file_path)

    if not torch.cuda.is_available():
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'gpu'}))
    else:
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'cpu'}))
   
    E.eval()

    train_dsets = Craft3DDataset(params.data_dir, "train") 
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=1, shuffle=False, num_workers=1)

    for i, X in enumerate(tqdm(train_dset_loaders)):
        
        z = E(X)
        z = z.detach().cpu().numpy()
        z_save_pth = os.path.join(embeddings_file_path, "%d.npy" % i)
        np.save(z_save_pth, z)


def tester_ae(args):
    print ('Evaluation Mode...')

    if args.debug:
        import pdb; pdb.set_trace()
        num_workers = 0
    else:
        num_workers = 1

    image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path+'/'+'G.pth'
    pretrained_file_path_E = save_file_path+'/'+'E.pth'
    
    G = net_G_blocks(args)
    E = net_E(args)

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'gpu'}))
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'gpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'cpu'}))
   
    G.eval()
    E.eval()

    train_dsets = Craft3DDatasetAnno(
        params.data_dir, "train", regress_parts=params.use_block_part,
        regress_types=params.use_block_type,
        only_popular_parts=params.only_popular_parts,
        part_augment=params.part_augment)
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=4, shuffle=True, num_workers=1)

    iou_avg, acc_avg = 0, 0

    for i, X in enumerate(tqdm(train_dset_loaders)):
        
        z = E(X)
        recon = G(z)

        recon = recon.detach().cpu().numpy()
        visualize_schematic_recon(X, recon, image_saved_path, i)

        """
        recon = (recon > 0.5).float().unsqueeze(0)
        X = (X > 0).float()
        acc_avg += (occupancy_accuracy(X, recon) - acc_avg) / (i+1)
        iou_avg += (occupancy_iou(X, recon) - iou_avg) / (i+1)
        print(acc_avg, iou_avg)

        # format schematic
        samples = samples[0].__ge__(0.3)
        x, y, z = samples.shape
        num_blocks = len(samples.nonzero()[0])
        block_array = np.concatenate(
            (np.ones((num_blocks, 1)), np.zeros((num_blocks, 1))),
            axis=1
        )

        schematic = np.zeros((x, y, z, 2))
        schematic[samples.nonzero()] = block_array
        schem_dict = {'name':['patio'],'tags':['patio'], 'schematic':schematic}

        schem_save_path = os.path.join(params.images_dir, "patio_%d.pkl" % i)
        with open(schem_save_path, 'wb') as f:
            pkl.dump(schem_dict, f)
        """
    print("Voxel Accuracy: %f \n Voxel IoU: %f" % (acc_avg, iou_avg))

def embedding_arithmetic(args):
    print ('Evaluation Mode...')

    image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    save_file_path = params.output_dir + '/' + args.model_name
    pretrained_file_path_G = save_file_path+'/'+'G.pth'
    pretrained_file_path_E = save_file_path+'/'+'E.pth'
    embeddings_file_path = save_file_path + "/embeddings/"
    
    G = net_G(args)
    E = net_E(args)
    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'gpu'}))
        E.load_state_dict(torch.load(pretrained_file_path_E, map_location={'cuda:0': 'gpu'}))
    G.eval()
    E.eval()

    train_dsets = Craft3DDataset(params.data_dir, "train") 
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=1, shuffle=False, num_workers=1)

    # setting up faiss
    index = faiss.IndexFlatL2(params.z_dim)
    for i in glob.glob(embeddings_file_path + '/*.npy'):
        v = np.expand_dims(np.load(i), 0)
        index.add(v)

    for i, X in enumerate(tqdm(train_dset_loaders)):
        
        z_s, X_hat_s = [], []

        z0 = E(X)
        z0 = np.expand_dims(z0.detach().cpu().numpy(), 0)
        # similarity search
        k = 2
        z1 = index.search(z0, k)[1][0][1]
        z1 = index.reconstruct(int(z1))

        """
        for oom in [1e-5, 1e-4, 1e-3, 1e-2]:
            noise = np.random.sample(params.z_dim) * oom
            noise = torch.from_numpy(noise).type(torch.float32)
            z_s.append(z + noise)

        for z in z_s:
            fake = G(z)
            samples = fake.detach().unsqueeze(0).cpu().numpy()
            X_hat_s.append(samples)
        """
        """
        z0 = z[0].unsqueeze(0)
        z_diff = z[1] - z[0]
        z_diff = z_diff.unsqueeze(0)
        for alpha in [0, .25, .5, .75]:
            z_interp = z0 + (alpha*z_diff)
            fake = G(z_interp)
            samples = fake.detach().unsqueeze(0).cpu().numpy()
            X_hat_s.append(samples)
        """

        z_diff = z1 - z0
        for alpha in [0, .25, .5, .75, 1]:
            z_interp = z0 + (alpha*z_diff)
            fake = G(torch.from_numpy(z_interp))
            samples = fake.detach().unsqueeze(0).cpu().numpy()
            X_hat_s.append(samples)

        #X_hat_s.append(X[1].unsqueeze(0))
        X = X[0].unsqueeze(0)

        ### visualization
        visualize_schematic_latent_noise(X, X_hat_s, image_saved_path, i)

        """
        # format schematic
        samples = samples[0].__ge__(0.3)
        x, y, z = samples.shape
        num_blocks = len(samples.nonzero()[0])
        block_array = np.concatenate(
            (np.ones((num_blocks, 1)), np.zeros((num_blocks, 1))),
            axis=1
        )

        schematic = np.zeros((x, y, z, 2))
        schematic[samples.nonzero()] = block_array
        schem_dict = {'name':['patio'],'tags':['patio'], 'schematic':schematic}

        schem_save_path = os.path.join(params.images_dir, "patio_%d.pkl" % i)
        with open(schem_save_path, 'wb') as f:
            pkl.dump(schem_dict, f)
        """

def tester_embedding_probe(args):

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
    train_dset_loader2 = torch.utils.data.DataLoader(
        train_dsets, batch_size=3, shuffle=False, num_workers=num_workers)

    for i, (ref_pre, ref_post) in enumerate(tqdm(train_dset_loader1)):

        #ref_pre.to(params.device)
        #ref_post.to(params.device)
        zref_pre = E(ref_pre)
        zref_post = E(ref_post)

        # removal
        # zref_diff = zref_post - zref_pre

        # addition
        zref_diff = zref_pre - zref_post

        ref_hat = G(zref_pre)[0].unsqueeze(0)

        iou_avg, acc_avg, roof_avg = 0, 0, 0

        for j, (X_pre, X_post) in enumerate(tqdm(train_dset_loader2)):
            
            #X_pre.to(params.device)
            #X_post.to(params.device)
            zX_pre, zX_post = E(X_pre), E(X_post)
    
            # removal
            # zhat = zX_pre + zref_diff

            # addition
            zhat = zX_post + zref_diff

            Xhat = G(zhat)[0]
            if len(Xhat.shape) < 4:
                Xhat = Xhat.unsqueeze(0)

            if j % 32 == 0:
                voxels = torch.cat((ref_post, X_post, ref_hat, Xhat))
                voxels = voxels.detach().cpu().numpy()
                visualize_schematic(voxels, image_saved_path, j+i)

            acc_avg += (occupancy_accuracy(X_post, Xhat).detach().cpu() - acc_avg) / (j+1)
            iou_avg += (occupancy_iou(X_post, Xhat).detach().cpu() - iou_avg) / (j+1)
            roof_count = len((X_pre != X_post).nonzero())
            if roof_count == 0:
                roof_count = 1
            roof_acc = Xhat[X_pre != X_post].sum() / roof_count
            roof_acc = roof_acc.detach().cpu()
            roof_avg += (roof_acc - roof_avg) / (j+1)

        print("Accuracy avg: %f, IoU avg: %f, Roof acc: %f" % (acc_avg, iou_avg, roof_avg))
            
