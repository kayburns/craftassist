"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import torch
import pickle
import torch.optim as optim
import torch.nn as nn
from data_loaders import make_example_from_raw
from train_semantic_segmentation import get_loss, online_update
import logging

class SemSegNet(nn.Module):
    def __init__(self, opts, classes=None):
        super(SemSegNet, self).__init__()

        if opts.load:
            if opts.load_model != "":
                self.load(opts.load_model)
            else:
                raise ("loading from file specified but no load_filepath specified")

            if opts.vocab_path != "":
                self.load_vocab(opts.vocab_path)
            else:
                self.vocab = None
                print ("loading from file specified but no vocab_path specified")
        else:
            self.opts = opts
            self._build()
            self.classes = classes
        self.online_convs = []
        self.online_classes = []

    def _build(self):
        opts = self.opts
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 8
        try:
            num_words = opts.num_words
        except:
            num_words = 3
        try:
            num_layers = opts.num_layers
        except:
            num_layers = 4  # 32x32x32 input
        try:
            hidden_dim = opts.hidden_dim
        except:
            hidden_dim = 64

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.out = nn.Conv3d(hidden_dim, opts.num_classes, kernel_size=1)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x, T=1):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(self.num_layers):
            z = self.layers[i](z)
        logits = self.out(z/T)
        if len(self.online_convs) > 0:
            online_logits = [out(z/T) for out in self.online_convs]
            
            # messy debugging; multiprocessing + pdb :'( </3
            #helpp, z = torch.max(online_logits[0], -1)
            #helpp, y = torch.max(helpp, -1)
            #max_value, x = torch.max(helpp, -1)
            #y = y[:,:,x]
            #z = z[:,:,x,y]
            #logging.info("ONLINE LOGITS: {}".format(max_value))
            #logging.info("SHAPE: {}".format(online_logits[0].shape))
            #logging.info("XYZ: {} {} {}".format(x, y, z))
            #logging.info("LOGITS: {}".format(logits[:, :, x, y, z]))
            logits = torch.cat([logits]+online_logits, axis=1)
        return self.lsm(logits)

    def fetch_initialization(self, x, y, T=1):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(self.num_layers):
            z = self.layers[i](z)
        init = (z * y).sum((-3, -2, -1)) / len(y.nonzero())
        init = init.reshape(init.shape[0], init.shape[1], 1, 1, 1)
        return init / (.3 * torch.norm(init))

    def save(self, filepath):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["classes"] = self.classes
        sds["state_dict"] = self.state_dict()
        torch.save(sds, filepath)
        if self.opts.cuda:
            self.cuda()

    def load_vocab(self, vocab_path):
        with open(vocab_path, "rb") as file:
            self.vocab = pickle.load(file)
        print("Loaded vocab")

    def load(self, filepath):
        sds = torch.load(filepath)
        self.opts = sds["opts"]
        print("loading from file, using opts")
        print(self.opts)
        self._build()
        self.load_state_dict(sds["state_dict"])
        self.zero_grad()
        self.classes = sds["classes"]

    def update_classes(self, cls):
        self.opts.num_classes += 1
        cls_idx = len(self.classes['idx2name'])
        self.classes['idx2name'].append(cls)
        self.classes['name2idx'][cls] = cls_idx
        self.classes['name2count'][cls] = 1 # does this matter?
        return cls_idx

    def add_class_online(self, cls, optimizer, init=None):
        name2idx = self.classes['name2idx']
        if cls not in name2idx.keys():
            self.online_convs.append(nn.Conv3d(self.hidden_dim, 1, kernel_size=1))
            if init is not None:
                self.online_convs[-1].weight = torch.nn.Parameter(init)
            optimizer.add_param_group({'params': self.online_convs[-1].parameters()})
            self.online_classes.append(cls)
            return self.update_classes(cls)
        else:
            return name2idx[cls]


class Opt:
    pass


class SemSegWrapper:
    def __init__(self, model, vocab_path, threshold=-1.0, blocks_only=True, cuda=False):
        if type(model) is str:
            opts = Opt()
            opts.load = True
            opts.load_model = model
            opts.vocab_path = vocab_path
            model = SemSegNet(opts)
        self.model = model
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        self.classes = model.classes
        # threshold for relevance; unused rn
        self.threshold = threshold
        # if true only label non-air blocks
        self.blocks_only = blocks_only
        # this is used by the semseg_process
        i2n = self.classes["idx2name"]
        self.tags = [(c, self.classes["name2count"][c]) for c in i2n]
        assert self.classes["name2idx"]["none"] == 0

        # for online updates
        self.loss = nn.NLLLoss(reduction="none")
        # key error: https://github.com/pytorch/pytorch/pull/17679
        # self.optimizer = optim.Adagrad(model.parameters(), lr=.01)
        self.optimizer = optim.SGD(model.parameters(), lr=.001) 
        if self.cuda:
            self.loss.cuda()

    @torch.no_grad()
    def segment_object(self, blocks, T=1):
        self.model.eval()

        if self.model.vocab:
            logging.info("USING VOCAB")
            vocab = self.model.vocab
            vocab_blocks = np.zeros(blocks.shape[:-1])
            for x in range(blocks.shape[0]):
                for y in range(blocks.shape[1]):
                    for z in range(blocks.shape[2]):
                        block_id = blocks[x,y,z,0]
                        meta_id = blocks[x,y,z,1]
                        id_tuple = (block_id, meta_id)
                        # First see if that specific block-meta pair is in the vocab.
                        if id_tuple in vocab:
                            id_ = vocab[id_tuple]
                        # Else, check if the same general material (block-id) exists.
                        elif (block_id, 0) in vocab:
                            id_ = vocab[(block_id, 0)]
                        # If not, the network has no clue what it is, ignore it (treat as air).
                        else:
                            id_ = vocab[(0,0)]
                        
                        vocab_blocks[x,y,z] = id_
        else:
            logging.info("NOT USING VOCAB")
            vocab_blocks = blocks[:, :, :, 0]

        blocks = torch.from_numpy(vocab_blocks)
        blocks, _, o = make_example_from_raw(blocks)
        logging.info("OOOOOOOOOOOO: {}".format(o))
        blocks = blocks.unsqueeze(0)
        if self.cuda:
            blocks = blocks.cuda()
        y = self.model(blocks, T=T)
        _, mids = y.squeeze().max(0)
        logging.info("IS WUG DETECTED: {}".format((mids==30).any()))
        try:
            logging.info("IS THIS YOUR CARD: {}, {}".format(y[0, :, 9, 6, 14], blocks[0, 9, 6, 14]))
        except:
            logging.info("NO CARD FOR YOU")
        locs = mids.nonzero()
        locs = locs.tolist()
        if self.blocks_only:
            return {
                tuple(np.subtract(l, o)): mids[l[0], l[1], l[2]].item()
                for l in locs
                if blocks[0, l[0], l[1], l[2]] > 0
            }
        else:
            return {tuple(ll for ll in l): mids[l[0], l[1], l[2]].item() for l in locs}

    def _update_tags(self):
        # this is used by the semseg_process
        self.classes = self.model.classes
        i2n = self.classes["idx2name"]
        self.tags = [(c, self.classes["name2count"][c]) for c in i2n]

    def update(self, label, blocks, house):

        x = house[:, :, :, 0]
        y = blocks[:, :, :, 0]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x, y, o = make_example_from_raw(x, labels=y)
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        online_update(x, y, label, self.model, self.loss, self.optimizer)
        self._update_tags()
        logging.info(self.tags)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=16, help="size of blockid embedding")
    parser.add_argument("--num_words", type=int, default=256, help="number of blocks")
    parser.add_argument("--num_classes", type=int, default=20, help="number of blocks")

    args = parser.parse_args()

    N = SemSegNet(args)
