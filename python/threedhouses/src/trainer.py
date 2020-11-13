'''
trainer.py

Train 3dgan models
'''

import torch
from torch import optim
from torch import  nn
from torch.nn import functional as F
from utils import *
import os

from model import net_G, net_D, net_E, net_G_blocks

# added
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import params
from tester import occupancy_iou, occupancy_accuracy
from tqdm import tqdm
import wandb

from datasets import Craft3DDataset, Craft3DPartsDataset, Craft3DDatasetAnno


def write_params(p, fname):
    ignore = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', \
        '__file__', '__cached__', '__builtins__', 'torch', 'print_params']
    with open(fname, 'w+') as f:
        for k, v in p.__dict__.items():
            if k not in ignore:
                f.write(str(k) + ' : '+ str(v) + '\n')

def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value
        
    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)

def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value
        
    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def iterate_through_dataset(args):

    if args.debug:
        import pdb; pdb.set_trace()

    #train_dsets = Craft3DDatasetAnno(params.data_dir, "train")
    train_dsets = Craft3DDataset(params.data_dir, "valid", remove=['none'], block_dropout=[0])
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=2, shuffle=False, num_workers=0)

    over_256 = 0
    for i, X in enumerate(tqdm(train_dset_loaders)):
        if torch.max(X) >= 256:
            over_256 += 1
            print(over_256)
        #examples = X.cpu().data.squeeze().numpy()
        #image_saved_path = params.images_dir
        
        #if not os.path.exists(image_saved_path):
        #    os.makedirs(image_saved_path)

        #visualize_schematic(examples, image_saved_path, i, labels)
               
def trainer_autoencoder(args):

    if args.debug:
        shuffle = False
        num_workers = 0
        import pdb; pdb.set_trace()
    else:
        num_workers = 1
        shuffle = True

  
    # added for output dir
    save_file_path = params.output_dir + '/' + args.model_name
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    write_params(params, os.path.join(save_file_path, 'params.txt'))

    if args.logs:
        wandb.init(project="houses")

    # datset define
    train_dsets = Craft3DDatasetAnno(
            params.data_dir, "train", params.noise,
            only_popular_parts=params.only_popular_parts,
            part_augment=params.part_augment)
    
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=params.batch_size, shuffle=shuffle,
        num_workers=num_workers)
    
    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}

    # model define
    E = net_E(args)
    G = net_G(args)

    print(dset_len)

    E_solver = optim.Adam(E.parameters(), lr=params.e_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    def lr_E(step):
        return params.e_lr
    def lr_G(step):
        return params.g_lr
    E_scheduler = optim.lr_scheduler.LambdaLR(E_solver, lr_lambda=lr_E)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_solver, lr_lambda=lr_G)

    E.to(params.device)
    G.to(params.device)

    criterion = nn.MSELoss()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):

        start = time.time()
        
        for phase in ['train']:
            if phase == 'train':
                E.train()
                G.train()
            else:
                E.eval()
                G.eval()

            running_loss = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                batch = X.size()[0]

                # ============= Train =============#
                encoded = E(X)
                recon = G(encoded)

                loss = criterion(X, recon)

                E.zero_grad()
                G.zero_grad()
                loss.backward()

                G_solver.step()
                E_solver.step()

                E_scheduler.step()
                G_scheduler.step()

                # =============== logging each 10 iterations ===============#

                running_loss += loss.item() * X.size(0)

                if args.logs and itr_train % 10 == 0:
                    wandb.log({'loss':loss})
                        
            # =============== each epoch save model or save image ===============#

            epoch_loss = running_loss / dset_len[phase]

            end = time.time()
            epoch_time = end - start


            print('Epochs-{} ({}) , Loss : {:.4}'.format(epoch, phase, epoch_loss))
            print ('Elapsed Time: {:.4} min'.format(epoch_time/60.0))

            if (epoch) % params.model_save_step == 0:

                print ('model_saved, images_saved...')
                torch.save(G.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'G' + '.pth')
                torch.save(E.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'E' + '.pth')

                samples = recon.cpu().data[:4].squeeze().numpy()
                image_saved_path = params.images_dir
                if not os.path.exists(image_saved_path):
                    os.makedirs(image_saved_path)

                SaveRecon_Voxels(X.cpu().data[:4].numpy(), samples, image_saved_path, epoch)
 
def trainer_autoencoder_bl(args):

    if args.debug:
        shuffle = False
        num_workers = 0
        import pdb; pdb.set_trace()
    else:
        num_workers = 1
        shuffle = True

    save_file_path = params.output_dir + '/' + args.model_name
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    write_params(params, os.path.join(save_file_path, 'params.txt'))

    train_dsets = Craft3DDatasetAnno(
        params.data_dir, "train", params.noise, params.use_block_part,
        params.use_block_type, params.only_popular_parts, params.part_augment)
    
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=params.batch_size, shuffle=shuffle,
        num_workers=num_workers)
    
    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}

    # model define
    E = net_E(args)
    G = net_G_blocks(args)
    if params.from_weights:
        E_w = torch.load(os.path.join(params.from_weights,args.model_name+'/E.pth'))
        G_w = torch.load(os.path.join(params.from_weights,args.model_name+'/G.pth'))
        E.load_state_dict(E_w)
        G.load_state_dict(G_w)

    if args.logs:
        wandb.init(project="houses")
        wandb.config.dataset = "houses"
        wandb.config.model = "ae"
        wandb.watch(E)
        wandb.watch(G)

    print(dset_len)

    E_solver = optim.Adam(E.parameters(), lr=params.e_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    def lr_E(step):
        return params.e_lr
    def lr_G(step):
        return params.g_lr
    E_scheduler = optim.lr_scheduler.LambdaLR(E_solver, lr_lambda=lr_E)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_solver, lr_lambda=lr_G)

    E.to(params.device)
    G.to(params.device)

    criterion_MSE = nn.MSELoss()
    if params.use_block_type:
        criterion_CE = nn.CrossEntropyLoss()
    if params.vae:
        criterion_KL = KLNormal()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):

        start = time.time()
        
        for phase in ['train']:
            if phase == 'train':
                E.train()
                G.train()
            else:
                E.eval()
                G.eval()

            running_loss = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):
                if torch.max(X) >= 256:
                    print("Skipped one batch")
                    continue

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                batch = X.size()[0]

                # ============= Train =============#
                if params.vae:
                    z = E(X)
                    m, z = z[:,:200], z[:,200:]
                    v = F.softplus(z) + 1e-8
                    encoded = sample_gaussian(m, v)
                else:
                    encoded = E(X)
 
                cls, reg = G(encoded)

                idx = X.nonzero(as_tuple=True)
                reg = reg.permute(0, 2, 3, 4, 1)

                loss_coords = criterion_MSE((X > 0).float(), cls)
                loss = loss_coords
                if params.use_block_type:
                    loss_types = criterion_CE(reg[idx], X[idx].long())
                    loss += (.0001 * loss_types)
                if params.vae:
                    m_target, v_target = torch.zeros_like(m), torch.ones_like(v)
                    loss_kl = criterion_KL(m, v, m_target, v_target).mean()
                    loss += (0 * loss_kl)
                
                E.zero_grad()
                G.zero_grad()
                loss.backward()

                G_solver.step()
                E_solver.step()

                E_scheduler.step()
                G_scheduler.step()

                # =============== logging each 10 iterations ===============#

                running_loss += loss.item() * X.size(0)

                if args.logs and itr_train % 10 == 0:
                    wandb.log({'loss':loss})
                    wandb.log({'loss_coords': loss_coords})
                    if params.use_block_type:
                        wandb.log({'loss_types': loss_types})
                    if params.vae:
                        wandb.log({'loss_vae': loss_kl})
                        
            # =============== each epoch save model or save image ===============#

            epoch_loss = running_loss / dset_len[phase]

            end = time.time()
            epoch_time = end - start


            print('Epochs-{} ({}) , Loss : {:.4}'.format(epoch, phase, epoch_loss))
            print ('Elapsed Time: {:.4} min'.format(epoch_time/60.0))

            if (epoch) % params.model_save_step == 0:

                print ('model_saved, images_saved...')
                torch.save(G.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'G' + '.pth')
                torch.save(E.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'E' + '.pth')

                samples = cls.cpu().data[:4].squeeze().numpy().__ge__(0.5)
                reg = reg.cpu().data[:4].squeeze().numpy()
                reg[samples >= 0] = 0
                image_saved_path = params.images_dir
                if not os.path.exists(image_saved_path):
                    os.makedirs(image_saved_path)

                SaveRecon_Voxels(X.cpu().data[:4].numpy(), samples, image_saved_path, epoch)
                visualize_schematic_recon(X.cpu().data[:4].numpy(), samples, image_saved_path, epoch)

def trainer(args):

    if args.debug:
        shuffle = False
        num_workers = 0
        import pdb; pdb.set_trace()
    else:
        num_workers = 1
        shuffle = True

    # dataset loading
    train_dsets = Craft3DDataset(params.data_dir, "train")
    train_dset_loaders = torch.utils.data.DataLoader(
        train_dsets, batch_size=params.batch_size, shuffle=shuffle,
        num_workers=num_workers)
    val_dsets = Craft3DDataset(
            params.data_dir, "valid", remove=['none'], block_dropout=[0])
    val_dset_loaders = torch.utils.data.DataLoader(
            val_dsets, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    
    # define and load model
    E = net_E(args)
    G = net_G(args)
    if params.from_weights:
        E_w = torch.load(os.path.join(params.from_weights, 'E.pth'))
        G_w = torch.load(os.path.join(params.from_weights,args.model_name+'/G.pth'))
        E.load_state_dict(E_w)
        G.load_state_dict(G_w)

    # logging
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    write_params(params, os.path.join(params.output_dir, 'params.txt'))

    if args.logs:
        wandb.init(project="houses")
        wandb.config.dataset = "houses"
        wandb.config.model = "vae removals"
        wandb.config.kl_weight = params.kl_weight
        wandb.config.e_lr = params.e_lr
        wandb.config.g_lr = params.g_lr
        wandb.config.batch_size = params.batch_size
        wandb.config.recon_loss = "BCE"
        wandb.config.pos_weight = params.pos_weight
        wandb.watch(E)
        wandb.watch(G)

    # optim
    E_solver = optim.Adam(E.parameters(), lr=params.e_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)
    def lr_E(step):
        return params.e_lr
    def lr_G(step):
        return params.g_lr
    E_scheduler = optim.lr_scheduler.LambdaLR(E_solver, lr_lambda=lr_E)
    G_scheduler = optim.lr_scheduler.LambdaLR(G_solver, lr_lambda=lr_G)

    E.to(params.device)
    G.to(params.device)

    pos_weight = torch.ones([params.cube_len] * 3) * params.pos_weight
    criterion_BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(params.device))
    criterion_CE = nn.CrossEntropyLoss()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):
        itr_train += 1
        start = time.time()
        
        E.train()
        G.train()
            
        running_loss = 0.0

        for i, (X, remove, dropout_p) in enumerate(tqdm(train_dset_loaders)):

            X = X.to(params.device)
            batch = X.size()[0]

            # ============= Train =============#
            z = E(X)
            mu, logv = z[:,:200], z[:,200:]
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            encoded = eps * std + mu
            logits, cls = G(encoded)

            loss_coords = criterion_BCE(logits, (X > 0).float())
            loss = loss_coords
            if params.vae:
                loss_kl = torch.mean(-0.5 * torch.sum(1 + logv - mu ** 2 - logv.exp(), dim = 1), dim = 0)
                loss += (params.kl_weight * loss_kl)
                
            E_solver.zero_grad()
            G_solver.zero_grad()
            loss.backward()
            G_solver.step()
            E_solver.step()
            E_scheduler.step()
            G_scheduler.step()

            # =============== logging each 10 iterations ===============#

            running_loss += loss.item() * X.size(0)

            if args.logs and itr_train % 10 == 0:
                wandb.log({'loss':loss})
                wandb.log({'loss_coords': loss_coords})
                if params.vae:
                    wandb.log({'loss_vae': loss_kl})
                        
        # =============== each epoch save model or save image ===============#

        epoch_loss = running_loss / len(train_dsets)

        end = time.time()
        epoch_time = end - start

        vis_X = X
        vis_cls = cls
        # val loop
        iou = acc = 0
        for i, (X, _, _) in enumerate(tqdm(val_dset_loaders)):
            X = X.to(params.device)
            batch = X.size()[0]

            z = E(X)
            mu, logv = z[:200], z[200:]
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            encoded = eps * std + mu
            _, cls = G(encoded)

            cls = cls.cpu().data.numpy().__ge__(0.5).astype(float)
            X = (X > 0).float().cpu().numpy()
            acc += occupancy_accuracy(X, cls)
            iou += occupancy_iou(X, cls)

        acc = acc / len(val_dsets)
        iou = iou / len(val_dsets)
        if args.logs:
            wandb.log({'val acc': acc})
            wandb.log({'val iou': iou})
 
        print('Epochs-{} ({}) , Loss : {:.4}'.format(epoch, "train", epoch_loss))
        print('Epochs-{} ({}) , Acc : {:.4}'.format(epoch, "val", acc))
        print('Epochs-{} ({}) , IoU : {:.4}'.format(epoch, "val", iou))
        print('Elapsed Time: {:.4} min'.format(epoch_time/60.0))

        if (epoch) % params.model_save_step == 0:

            print ('model_saved, images_saved...')
            torch.save(G.state_dict(), params.output_dir + '/G.pth')
            torch.save(E.state_dict(), params.output_dir + '/E.pth')

            vis_cls = vis_cls.cpu().data[:4].squeeze().numpy().__ge__(0.5)
            vis_X = vis_X.cpu().data[:4].numpy()
            image_saved_path = params.images_dir
            if not os.path.exists(image_saved_path):
                os.makedirs(image_saved_path)

            visualize_schematic_recon(vis_X, vis_cls, image_saved_path, epoch)
 
