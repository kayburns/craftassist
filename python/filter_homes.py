import os
import glob
import argparse

import numpy as np

ANNOYING_BLOCKS = [8, 10] # things like flowing water and lava behave weirdly in minecraft

def contains_no_annoying_blocks(house):
    return not np.any(np.isin(house, ANNOYING_BLOCKS))

def contains_blocks(house, blocks):
    """
    blocks: list of block ids
    """
    return np.any(np.isin(house, blocks))

def remove_annoying_blocks(house):
    # TODO
    return

def is_small_enough(house):
    x, y, z = house.shape[:3]
    return x < 15 and y < 15 and z < 15

def fix_homes(dir_name, filter_funcs=[]):
    """Rewrites validation files after applying filter_funcs"""
    # TODO
    return

def filter_homes(f_pattern, filter_funcs=[is_small_enough]):
    """Returns list of files for use in experiments."""
    fnames = glob.glob(f_pattern)
    filtered_homes = []
    for fname in fnames:
        home = np.load(fname)
        well_formed = [func(home) for func in filter_funcs]
        if all(well_formed):
            filtered_homes.append(fname)
    return filtered_homes


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

    filtered_homes = filter_homes(args.f_pattern, filter_funcs=[lambda x: contains_blocks(x, [19])])
    if args.out_file:
        with open(args.out_file, 'w') as f:
            for home in filtered_homes:
                f.write(f"{home}\n")
    print(len(filtered_homes))

