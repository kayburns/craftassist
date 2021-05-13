import os
import glob
import random
import argparse

import numpy as np

ANNOYING_BLOCKS = [8, 10] # things like flowing water and lava behave weirdly in minecraft
with open('/craftassist/val_split.txt', 'r') as f:
    VAL_SPLIT = [l.strip() for l in f.readlines()]

def contains_no_annoying_blocks(house, name):
    return not np.any(np.isin(house, ANNOYING_BLOCKS))

def contains_blocks(house, blocks):
    """
    blocks: list of block ids
    """
    return np.any(np.isin(house, blocks))

def remove_annoying_blocks(house):
    # TODO
    return

def is_small_enough(house, name):
    x, y, z = house.shape[:3]
    # x is height
    return x < 12 and y < 21 and z < 21

def is_big_enough(house, name):
    x, y, z = house.shape[:3]
    return x > 6 and y > 6 and z > 6

def fix_homes(dir_name, filter_funcs=[]):
    """Rewrites validation files after applying filter_funcs"""
    # TODO
    return

def in_test(house, name):
    return name in VAL_SPLIT

def filter_homes(f_pattern, filter_funcs=[is_small_enough, in_test]):
    """Returns list of files for use in experiments."""
    fnames = glob.glob(f_pattern)
    filtered_homes = []
    for fname in fnames:
        home = np.load(fname)
        well_formed = [func(home, fname) for func in filter_funcs]
        if all(well_formed):
            filtered_homes.append(fname)
    return filtered_homes[:len(filtered_homes)//2]

# run with no filter funcs to create consistent test set
def split(f_pattern, filter_funcs=[]):
    """Returns list of files for use in experiments."""
    fnames = glob.glob(f_pattern)
    random.shuffle(fnames)
    filtered_homes = []
    for fname in fnames:
        home = np.load(fname)
        well_formed = [func(home) for func in filter_funcs]
        if all(well_formed):
            filtered_homes.append(fname)
    return filtered_homes[:len(filtered_homes)//2]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f_pattern', type=str,
        default='/craftassist/minecraft_specs/schematics/cleaned_houses/validation*.npy'
    )
    parser.add_argument(
        '--out_file',
        help='txt file to output filtered homes to, if desired'
    )
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()

    filtered_homes = filter_homes(
        args.f_pattern,
        filter_funcs=[
            contains_no_annoying_blocks,
            in_test,
            is_small_enough,
            is_big_enough
        ]
    )
    if args.out_file:
        with open(args.out_file, 'w') as f:
            for home in filtered_homes:
                f.write(f"{home}\n")
    print(len(filtered_homes))

