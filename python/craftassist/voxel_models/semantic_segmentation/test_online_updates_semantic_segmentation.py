"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import argparse
import sys
import random
from tqdm import tqdm
from data_loaders import SemSegData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import semseg_models as models
from train_semantic_segmentation import *
from pathlib import Path



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
    parser.add_argument("--batchsize", type=int, default=4, help="batch size")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--vocab_path", default="")
    parser.add_argument(
        "--save_model",
        default="./sem_seg_one_shot.pth",
        help="where to save model (nowhere if blank)"
    )
    parser.add_argument(
        "--load_model",
        default="/craftassist/python/craftassist/models/vision/sem_seg_model.pth",
        help="from where to load model (nowhere if blank)"
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

    to_keep = ["ceiling light", "entryway", "houseplant", "house support", "seat"]
    train_data_prev = SemSegData(
        data_dir / "training_data.pkl",
        nexamples=args.debug,
        augment=aug
    )
    valid_data_prev = SemSegData(
        data_dir / "validation_data.pkl", 
        classes_to_match=train_data_prev.classes
        )

    train_data = SemSegData(
        data_dir / "training_data.pkl",
        nexamples=args.debug,
        augment=aug,
        to_keep=to_keep
    )
    valid_data = SemSegData(
        data_dir / "validation_data.pkl", 
        classes_to_match=train_data.classes
        )



    shuffle = True
    if args.debug > 0:
        shuffle = False

    def make_dataloader(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batchsize,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
            num_workers=args.ndonkeys,
        )

    valid_dl_prev = make_dataloader(valid_data_prev) 

    args.num_classes = len(train_data.classes["idx2name"]) - len(to_keep)
    print("making model")
    args.load = False
    if args.load_model != "":
        args.load = True
    model = models.SemSegNet(args, classes=train_data.classes)
    nll = nn.NLLLoss(reduction="none")

    if args.cuda:
        model.cuda()
        nll.cuda()

    import pdb; pdb.set_trace()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # run validation before
    #valid_losses, valid_accs = validate(model, valid_dl_prev, nll, args, "entryway")
    #print(f"Valid acc before training: {sum(valid_accs) / len(valid_accs)}")

    samples = []
    for k in to_keep:
        samples.extend([train_data.get_all_x_with_class(k)[0]])
    random.shuffle(samples)
    
    for i, (cls, cidx, b) in enumerate(samples):
        print("Updating sample {} out of {}".format(i+1, len(samples)))
        model.train()
        x, y = b[0], b[1]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y = y == cidx # convert type?
        online_update(x, y, cls, model, nll, optimizer)

        # evaluate recall of single example
        model.eval()
        with torch.no_grad():
            yhat = model(x).max(1).indices
            yhat[x == 0] = 0
            cidx = model.classes['name2idx'][cls]
            print("IoU on updated sample: {}".format(
                get_single_class_iou(y*cidx, yhat, cidx)
            ))

    # run validation after
    for k in to_keep:
        valid_dl = train_data.get_all_x_with_class(k)
        valid_dl = [(b[0].unsqueeze(0),b[1].unsqueeze(0)) for _, _, b in valid_dl]
        import pdb; pdb.set_trace()
        valid_losses, valid_accs = validate_iou(model, valid_dl, nll, args)
        print("Valid acc on homes with {}: {} / {}".format(k, sum(valid_accs), len(valid_accs)))

    if args.save_model != "":
        model.save(args.save_model)

