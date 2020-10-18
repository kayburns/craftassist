import torch
import torch.nn as nn
import numpy as np
import faiss
import pickle as pkl
import os

class DecomposeOnline():

    def __init__(self, dim=768, db_file="db.pkl", thres=20):
        self.index = faiss.IndexFlatL2(dim)
        self.thres = thres
        if os.path.exists(db_file):
            xb, seqs = pkl.load(db_file)
            self.index.add(xb)
            self.seqs = seqs
        else:
            self.seqs = []
        print(self.index.ntotal)

    def reduce_word_features(self, word_feats):
        return np.average(word_feats, axis=1)

    def maybe_get_parse(self, x_reps):
        x_reps = x_reps.cpu().detach().numpy()
        feat = self.reduce_word_features(x_reps)
        D, I = self.index.search(feat, k=1)
        if D[0][0] > self.thres:
            return None
        else:
            return self.seqs[I[0][0]]

    def update(self, x_reps, action_sequence):
        x_reps = x_reps.cpu().detach().numpy()
        feat = self.reduce_word_features(x_reps)
        self.index.add(feat)
        self.seqs.append(action_sequence)

    def save(self):
        return None

