#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import tarfile
import warnings
import itertools
import csv
import json
from os import path as osp
from typing import Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pickle as pkl
import requests
import torch
from torch.utils.data import Dataset

class Craft3DDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        subset: str,
        noise: list = [0],
        regress_parts: bool = False,
        regress_types: bool = False,
        only_popular_parts: bool = False,
        part_augment: bool = False,
        remove: str = None
   ):
        super().__init__()
        self.subset = subset
        self.data_dir = data_dir

        # raw training data
        fname = ""
        if subset == "train":
            fname = "training_data.pkl"
        elif subset == "valid":
            fname = "validation_data.pkl"
        fname = osp.join(self.data_dir, fname)
        self.raw_items = self._load_raw(fname)
        
        # references to raw training data, with various augmentations specified
        self.items = self._create_items()

    def __getitem__(self, index):
        raw_idx = self.items[index]
        return

    def __len__(self):
        return len(self.items)
    
    def _load_raw(self, fname):
        if not osp.isfile(fname):
            raise RuntimeError(f"Split file not found at: {fname}")
        with open(fname, "rb") as f:
            raw_items = pkl.load(f)

        standardized_items = []
        for item in raw_items:
            import pdb; pdb.set_trace()
            schematic = torch.from_numpy(item[0])
            schematic = self._standardize(schematic.permute(1, 2, 0))
            part_schem = torch.from_numpy(item[1])
            part_schem = self._standardize(part_schem.permute(1, 2, 0))
            standardized_items.append((schematic, part_schem))

        return raw_items

    def _create_items(self):
        return

    def _standardize(self, annotation, noise=0):
        standardized = torch.zeros(64, 64, 64)
        x, y, z = annotation.shape
        # centering with noise
        noise_y = min(32 - (y//2) - 1, noise)
        noise_z = min(32 - (z//2) - 1, noise)
        y_idx = max(0, 32 - (y//2)+noise_y)
        z_idx = max(0, 32 - (z//2)+noise_z)
        standardized[:x, y_idx:y_idx+y, z_idx:z_idx+z] = annotation
        return standardized


class Craft3DDatasetAnno(Dataset):

    def __init__(
        self,
        data_dir: str,
        subset: str,
        noise: list = [0],
        regress_parts: bool = False,
        regress_types: bool = False,
        only_popular_parts: bool = False,
        part_augment: bool = False,
        remove: str = None
   ):
        super().__init__()
        self.data_dir = data_dir
        self.max_dim = 64
        self.subset = subset
        self.regress_parts = regress_parts
        self.regress_types = regress_types
        self.only_popular_parts = only_popular_parts
        self.part_augment = part_augment
        self.remove = remove

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        self.blockname2id = self.create_blockname2id()
        self.id2blockname = dict((reversed(b2id) for b2id in self.blockname2id.items()))
        self.block_type_remap = self.create_block_type_remap()
        self.blockid2clsid = self.read_blockid2clsid()
        #self.blockid2clsid = dict(zip(self.id2blockname.keys(),range(len(self.id2blockname))))
        self.unique_parts = defaultdict(lambda: 0)
        self.unique_types = defaultdict(lambda: 0)
        self._load_dataset()
        self.popular_parts = {k:v for (k,v) in self.unique_parts.items() if v>50}

    def __len__(self) -> int:
        """ Get number of valid blocks """
        return len(self._all_structures)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """ Get the index-th valid block
        """
        schem, seg_schem, part_labels, label, idxs = self._all_structures[index]

        annotation = schem
        if self.only_popular_parts:
            idxs = {k:v for (k, v) in part_labels.items() if k in self.popular_parts}
            idxs = sum(idxs.values(), [])
            annotation = self.select_substructure(schem, seg_schem, idxs)
        if not self.regress_types:
            annotation = (annotation > 0).float()
        if self.remove:
            idxs = {k:v for (k, v) in part_labels.items() if k == self.remove}
            idxs = sum(idxs.values(), [])
            a2 = self.remove_substructure(schem, seg_schem, idxs)
            a1, a2 = annotation.float(), (a2 > 0).float()
            return a1, a2

        return annotation.float()

    def print_statistics(self):
        # use to inspect block types
        blockname2count = {}
        for name, bid in self.blockname2id.items():
            if bid in self.unique_types:
                blockname2count[name] = self.unique_types[bid]
        blockname2count = {k: v for k, v in sorted(blockname2count.items(), key=lambda item: item[1])}
        with open('block_type_counts.txt', 'w+')as f:
            for k, v in blockname2count.items():
                f.write(str(k) + ','+ str(v) + '\n')
        #debug = [k for k in self.unique_types.keys() if not k in blockname2id.values()]
        #self.blockid2count = {blockname2id[k]:v for (k,v) in blockname2count.items()}
        #self.clsid2reducedclsid, i = {}, 0
        #for bid, count in self.blockid2count:
        #    if count > 410:
        #        self.clsid2reducedclsid[self.blockid2clsid[bid]] = i
        #        i += 1

        print("Number of unique types: %d" % len(self.unique_types))
        print("Number of unique parts: %d" % len(self.unique_parts))

    def _load_dataset(self):
        if self.subset == "train":
            split_fname = "training_data.pkl"
        else:
            split_fname = "validation_data.pkl"
        splits_path = osp.join(self.data_dir, split_fname)
        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "rb") as f:
            _all_houses = pkl.load(f)

        self._all_structures = []
        for house in _all_houses:
            schematic_all_blocks = torch.from_numpy(house[0])
            schematic = torch.zeros_like(schematic_all_blocks)
            part_schem = torch.from_numpy(house[1])

            for elem in torch.unique(schematic_all_blocks):
                schematic[schematic_all_blocks == elem] = self.blockid2clsid[elem.item()]

            # map instances to part labels
            part_labels = defaultdict(list)
            for i, label in enumerate(house[2][1:]):
                part_labels[label].append(i+1)
            for label in part_labels.keys():
                self.unique_parts[label] += 1

            # track unique block types
            block_type_ids = torch.unique(schematic)
            for blk_id in block_type_ids:
                blk_id = blk_id.item()
                self.unique_types[blk_id] += 1 

            if self.regress_parts:
                schematic = self.create_part_labels(part_schem, part_labels)

            print("warning! remapping ignored")
            schematic = schematic_all_blocks
            schematic = self._standardize(schematic.permute(1, 2, 0))
            part_schem = self._standardize(part_schem.permute(1, 2, 0))
            
            self._all_structures.append(
                (schematic, part_schem, part_labels, "house", list(range(256))))

            if self.part_augment:
                for label, idxs in part_labels.items():
                    self._all_structures.append(
                        (schematic, part_schem, part_labels, label, idxs))
        self.print_statistics()

    def create_part_labels(self, part_schem, part_labels):
        """Modifies part_schem in-place to create a new annotation with type
        annotation, as oppoosed to part annotations."""
        for _, idxs in part_labels.items():
            type_idx = idxs[0]
            for idx in idxs:
                part_schem[part_schem==idx] = type_idx
        return part_schem

    def create_blockname2id(self):
        blockname2id = {}
        block_map = os.path.join(self.data_dir, 'blocks.csv')
        with open(block_map, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['id2']) == 0: # only use canonical category
                    blockname2id[row['name']] = int(row['id1'])
        return blockname2id

    def create_block_type_remap(self):
        block_remap_names = {}
        block_remap = os.path.join(self.data_dir, "block_type_map.txt")
        with open(block_remap, "r") as f:
            for line in f:
                new_name, old_names = line.split("\"")[1], line.split("\"")[3]
                block_remap_names[new_name] = old_names.split(",")
        block_remap = {}
        for new_name, old_names in block_remap_names.items():
            old_ids = [self.blockname2id[n] for n in old_names]
            block_remap[self.blockname2id[new_name]] = old_ids
        return block_remap
        
    def read_blockid2clsid(self):
        fpath = os.path.join(self.data_dir, 'bid2clsid_500f_post_remap.txt')
        blockid2clsid = {}
        with open(fpath, "r") as f:
            for line in f:
                bid, clsid = line.split(',')
                blockid2clsid[int(bid)] = int(clsid)
        return blockid2clsid

    def create_blockid2clsid(self):
        idxs = self.blockname2id.values()
        remapped_idxs = self.block_type_remap.values()
        remapped_idxs = list(itertools.chain.from_iterable(remapped_idxs))
        remapped_idxs = [idx for idx in remapped_idxs if not idx in self.block_type_remap.keys()]
        idxs = [idx for idx in idxs if not idx in remapped_idxs]
        idxs = sorted(idxs)
        return dict(zip(idxs, range(len(idxs))))

    def select_substructure(self, schematic, part_schem, idxs):
        structure = torch.zeros_like(schematic)
        for idx in idxs:
            structure[part_schem == idx] = 1
        structure = structure * schematic
        return structure

    def remove_substructure(self, schematic, part_schem, idxs):
        structure = torch.ones_like(schematic)
        structure = structure * part_schem
        for idx in idxs:
            structure[part_schem == idx] = 0
        return structure

    def _standardize(self, annotation, noise=0):
        standardized = torch.zeros(64, 64, 64)
        x, y, z = annotation.shape
        # centering with noise
        noise_y = min(32 - (y//2) - 1, noise)
        noise_z = min(32 - (z//2) - 1, noise)
        y_idx = max(0, 32 - (y//2)+noise_y)
        z_idx = max(0, 32 - (z//2)+noise_z)
        standardized[:x, y_idx:y_idx+y, z_idx:z_idx+z] = annotation
        return standardized



class Craft3DDatasetStale(Dataset):
    NUM_BLOCK_TYPES = 256

    def __init__(
        self,
        data_dir: str,
        subset: str,
        noise: list = [0],
        use_block_type: bool = False
   ):
        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.max_dim = 64
        self.noise = noise
        self.use_block_type =  use_block_type

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        self._load_dataset_fnames()

    def __len__(self) -> int:
        """ Get number of valid blocks """
        return len(self._all_houses)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """ Get the index-th valid block

        Returns:
            A tuple of inputs and targets, where inputs is a dict of
            ```
            {
                "local": float tensor of shape (C * H, D, D, D),
                "global": float tensor of shape (1, G, G, G),
                "center": int tensor of shape (3,), the coordinate of the last block
            }
            ```
            where C is the number of block types, H is the history length, D is the
            local size, and G is the global size.

            targets is a dict of
            ```
            {
                "coords": int tensor of shape (A,)
                "types": int tensor of shape (A,)
            }
            ```
            where A is the number of next steps to be considered as targets.
        """
        annotation, n = self._all_houses[index]
        if isinstance(annotation, str):
            annotation = self._load_annotation(annotation)
            annotation[:,:,:,1] = 0 # zero out second block label
            annotation = annotation.sum(axis=-1)
            if not self.use_block_type:
                annotation = annotation > 0
                annotation = annotation.int()
        else:
            annotation = self._create_prior(annotation)

        x, y, z = annotation.shape
                
        return self._standardize(annotation, noise=n)

    def _load_dataset_fnames(self):
        splits_path = osp.join(self.data_dir, "splits.json")
        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self._all_houses = []
        max_len = 0
        for filename in splits[self.subset]:
            annotation = osp.join(self.data_dir, "houses", filename, "schematic.npy")
            if not osp.isfile(annotation):
                warnings.warn(f"No annotation file for: {annotation}")
                continue
            loaded = self._load_annotation(annotation)
            if max(loaded.shape) <= self.max_dim and len(loaded.nonzero()):
                for n in self.noise:
                    self._all_houses.append((annotation, n))

        """
        specs = [list(range(10,20))] * 3
        specs = itertools.product(*specs)
        for spec in specs:
            x, y, z = spec
            for n in self.noise:
                self._all_houses.append(((z, x, y), n))
        """

    def _load_dataset(self):
        splits_path = osp.join(self.data_dir, "splits.json")
        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self._all_houses = []
        max_len = 0
        for filename in splits[self.subset]:
            annotation = osp.join(self.data_dir, "houses", filename, "schematic.npy")
            if not osp.isfile(annotation):
                warnings.warn(f"No annotation file for: {annotation}")
                continue
            annotation = self._load_annotation(annotation)
            if max(annotation.shape) <= self.max_dim and len(annotation.nonzero()):
                annotation = annotation.sum(axis=-1)
                x, y, z = annotation.shape
                
                if False and z > y:
                    annotation = annotation.permute(0, 2, 1)
                    for n in self.noise:
                        self._all_houses.append(
                            self._standardize(annotation, noise=n))

                else:
                    # to-do: iterate twice
                    for n in self.noise:
                        self._all_houses.append(
                            self._standardize(annotation, noise=n))

                #standardized[x_idx:x_idx+x, :y, z_idx:z_idx+z] = annotation
                #standardized[:x, y_idx:y_idx+y, z_idx:z_idx+z] = annotation
                #standardized[x_idx:x_idx+x, y_idx:y_idx+y, :z] = annotation
                #self._all_houses.append(standardized)

    def _load_annotation(self, annotation_path: str) -> torch.Tensor:
        annotation = np.load(annotation_path)
        annotation = torch.from_numpy(annotation)
        return annotation

    def _standardize(self, annotation, noise=0):
        standardized = torch.zeros(64, 64, 64)
        x, y, z = annotation.shape
        # centering with noise
        noise_y = min(32 - (y//2) - 1, noise)
        noise_z = min(32 - (z//2) - 1, noise)
        y_idx = max(0, 32 - (y//2)+noise_y)
        z_idx = max(0, 32 - (z//2)+noise_z)
        standardized[:x, y_idx:y_idx+y, z_idx:z_idx+z] = annotation
        return standardized

    def create_part_labels(self, seg_schem, seg_labels):
        """Modifies seg_schem in-place to create a new annotation with type
        annotation, as oppoosed to part annotations."""
        for _, idxs in seg_labels.items():
            type_idx = idxs[0]
            for idx in idxs:
                seg_schem[seg_schem==idx] = type_idx
        return seg_schem

    def _create_prior(self, dim):
        z, x, y = dim
        schematic = np.zeros((z, x, y))
        schematic[:,0,:] = 1
        schematic[:,-1,:] = 1
        schematic[:,:,0] = 1
        schematic[:,:,-1] = 1
        schematic[-1,:,:] = 1
        schematic = torch.from_numpy(schematic)
        return schematic

class Craft3DPartsDataset(Dataset):
    NUM_BLOCK_TYPES = 256

    def __init__(
        self,
        data_dir: str,
        subset: str,
   ):
        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.max_dim = 64

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        self._load_dataset()

    def __len__(self) -> int:
        """ Get number of valid blocks """
        return len(self._all_houses)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """ Get the index-th valid block

        Returns:
            A tuple of inputs and targets, where inputs is a dict of
            ```
            {
                "local": float tensor of shape (C * H, D, D, D),
                "global": float tensor of shape (1, G, G, G),
                "center": int tensor of shape (3,), the coordinate of the last block
            }
            ```
            where C is the number of block types, H is the history length, D is the
            local size, and G is the global size.

            targets is a dict of
            ```
            {
                "coords": int tensor of shape (A,)
                "types": int tensor of shape (A,)
            }
            ```
            where A is the number of next steps to be considered as targets.
        """
        annotation = self._all_houses[index]
        return annotation

    def _load_dataset(self):
        files = os.listdir(self.data_dir)
        self._all_houses = []
        max_len = 0
        dims = []
        for filename in files:
            filename = os.path.join(self.data_dir, filename)
            annotation = self._load_annotation(filename)
            if max(annotation.shape) <= self.max_dim:
                standardized = torch.zeros(64, 64, 64)
                annotation = annotation.sum(axis=-1)
                x, y, z = annotation.shape
                standardized[:x, :y, :z] = annotation
                self._all_houses.append(standardized)
            dims.append(list(annotation.shape))

    def _load_annotation(self, annotation_path: str) -> torch.Tensor:
        annotation = np.load(annotation_path)
        annotation = torch.from_numpy(annotation)
        return annotation
