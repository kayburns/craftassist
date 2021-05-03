import os
import glob
from requests import get
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.nn as nna

class DecomposeOnline():

    def __init__(self, dim=768, db="/craftassist/db/", thres=20):
        self.ip = get('https://ifconfig.me').text
        self.index = faiss.IndexFlatL2(dim)
        self.thres = thres
        self.db = db
        self.seqs = []
        if os.path.exists(db):
            npy_files = glob.glob(os.path.join(self.db, '*.npy'))
            txt_files = [fname.replace('npy', 'txt') for fname in npy_files]
            for fname in npy_files:
                self.index.add(np.load(fname))
            for fname in txt_files:
                with open(fname, 'r') as f:
                    seq = f.readlines()
                    seq = [s.strip() for s in seq] 
                    self.seqs.append(seq[1:])
        else:
            self.seqs = []
        print(self.index.ntotal)

    def reduce_word_features(self, word_feats):
        return np.sum(word_feats, axis=1)

    def maybe_get_parse(self, x_reps):
        x_reps = x_reps.cpu().detach().numpy()
        feat = self.reduce_word_features(x_reps)
        D, I = self.index.search(feat, k=1)
        if D[0][0] > self.thres:
            return None
        else:
            return self.seqs[I[0][0]]

    def write_to_db(self, vec, new_command, seq):
        fname = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f") + f"_{self.ip}"
        # TODO: bad
        np.save(os.path.join(self.db, f'new/{fname}.npy'), vec)
        with open(os.path.join(self.db, f'new/{fname}.txt'), 'w') as f:
            f.write(f'{new_command}\n')
            for line in seq:
                f.write(f'{line}\n')
        np.save(os.path.join(self.db, f'{fname}.npy'), vec)
        with open(os.path.join(self.db, f'{fname}.txt'), 'w') as f:
            f.write(f'{new_command}\n')
            for line in seq:
                f.write(f'{line}\n')


    def update(self, x_reps, new_command, action_sequence):
        x_reps = x_reps.cpu().detach().numpy()
        feat = self.reduce_word_features(x_reps)
        self.index.add(feat)
        self.seqs.append(action_sequence)
        self.write_to_db(feat, new_command, action_sequence)
        
    def save(self):
        return None

