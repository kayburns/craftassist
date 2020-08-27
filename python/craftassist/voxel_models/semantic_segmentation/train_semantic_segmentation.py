"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import argparse
import sys
from tqdm import tqdm
from data_loaders import SemSegData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import semseg_models as models
from pathlib import Path


##################################################
# for debugging
##################################################


def print_slices(model, H, r, c, n, data):
    x, y = data[n]
    x = x.unsqueeze(0).cuda()
    yhat = model(x).squeeze()
    print(x[0, c - r : c + r, H, c - r : c + r].cpu())
    print(y[c - r : c + r, H, c - r : c + r])
    _, mm = yhat.max(0)
    print(mm[c - r : c + r, H, c - r : c + r].cpu())


def blocks_from_data(data, n):
    x, y = data[n]
    ids = x.nonzero()
    idl = ids.tolist()
    blocks = [((b[0], b[1], b[2]), (x[b[0], b[1], b[2]].item() + 1, 0)) for b in idl]
    return x, y, blocks


def semseg_output(S, n, data):
    x, y, blocks = blocks_from_data(data, n)
    class_stats = {}
    for i in range(29):
        class_stats[train_data.classes["idx2name"][i]] = len((y == i).nonzero())
        # print(train_data.classes['idx2name'][i], len((y==i).nonzero()))
    a = S._watch_single_object(blocks)
    return class_stats, a


##################################################
# training loop
##################################################


def get_loss(x, y, yhat, loss):
    # loss is expected to not reduce
    preloss = loss(yhat, y)
    mask = torch.zeros_like(y).float()
    u = x.float() + x.float().uniform_(0, 1)
    idx = u.view(-1).gt((1 - args.sample_empty_prob)).nonzero().squeeze()
    mask.view(-1)[idx] = 1
    M = float(idx.size(0))
    # FIXME: eventually need to intersect with "none" tags; want to push loss on labeled empty voxels
    preloss *= mask
    l = preloss.sum() / M
    return l


def get_accuracy(y, yhat):
    vals, pred = torch.max(yhat, 1)
    correct_num = torch.sum(pred == y)
    total_num = float(torch.numel(y))
    acc = correct_num / total_num
    return acc


def validate(model: nn.Module, validation_data: DataLoader, loss, args):
    losses = []
    accs = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(validation_data):
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            yhat = model(x)
            l = get_loss(x, y, yhat, loss)
            a = get_accuracy(y, yhat)
            accs.append(a.item())
            losses.append(l.item())
    return losses, accs
        

def train_epoch(model, DL, loss, optimizer, args):
    model.train()
    losses = []
    accs = []
    for b in tqdm(DL):
        x = b[0]
        y = b[1]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        model.train()
        yhat = model(x)

        l = get_loss(x, y, yhat, loss)
        a = get_accuracy(y, yhat)
        losses.append(l.item())
        accs.append(a.item())
        l.backward()
        optimizer.step()
    return losses, accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", type=int, default=-1, help="no shuffle, keep only debug num examples"
    )
    parser.add_argument("--num_labels", type=int, default=50, help="How many top labels to use")
    parser.add_argument("--num_epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--augment", default="none", help="none or maxshift:K_underdirt:J")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--save_model", default="", help="where to save model (nowhere if blank)")
    parser.add_argument(
        "--load_model", default="", help="from where to load model (nowhere if blank)"
    )
    parser.add_argument("--save_logs", default="/dev/null", help="where to save logs")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=4, help="size of blockid embedding")
    parser.add_argument("--lr", type=float, default=0.01, help="step size for net")
    parser.add_argument(
        "--sample_empty_prob",
        type=float,
        default=0.01,
        help="prob of taking gradients on empty locations",
    )

    parser.add_argument("--num_words", default=1024, type=int, help="number of rows in embedding table")

    parser.add_argument("--ndonkeys", type=int, default=4, help="workers in dataloader")
    args = parser.parse_args()

    if args.save_model == "":
        print("WARNING: No save path specified, model will not be saved.")

    this_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(this_dir, "../")
    sys.path.append(parent_dir)

    print("loading train data")
    aug = {}
    if args.augment != "none":
        a = args.augment.split("_")
        aug = {t.split(":")[0]: int(t.split(":")[1]) for t in a}
        aug["flip_rotate"] = True
    if args.debug > 0 and len(aug) > 0:
        print("warning debug and augmentation together?")
    
    data_dir = Path(args.data_dir)

    train_data = SemSegData(data_dir / "training_data.pkl", nexamples=args.debug, augment=aug)
    print("loaded train")
    valid_data = SemSegData(
        data_dir / "validation_data.pkl", 
        classes_to_match=train_data.classes, 
        )
    print("loaded valid")

    shuffle = True
    if args.debug > 0:
        shuffle = False

    print("making dataloader")
    
    def make_dataloader(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batchsize,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
            num_workers=args.ndonkeys,
        )
    
    rDL = make_dataloader(train_data)
    valid_dl = make_dataloader(valid_data)

    args.num_classes = len(train_data.classes["idx2name"])
    print("making model")
    args.load = False
    if args.load_model != "":
        args.load = True
    model = models.SemSegNet(args, classes=train_data.classes)
    nll = nn.NLLLoss(reduction="none")

    if args.cuda:
        model.cuda()
        nll.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    print("training")
    for m in tqdm(range(args.num_epochs)):
        train_losses, train_accs = train_epoch(model, rDL, nll, optimizer, args)
        valid_losses, valid_accs = validate(model, valid_dl, nll, args)
        print(f"\nEpoch {m}:")
        print(f"Train loss: {sum(train_losses) / len(train_losses)}")
        print(f"Valid loss: {sum(valid_losses) / len(valid_losses)}")
        print(f"Train acc: {sum(train_accs) / len(train_accs)}")
        print(f"Valid acc: {sum(valid_accs) / len(valid_accs)}")

        if args.save_model != "":
            model.save(args.save_model)
