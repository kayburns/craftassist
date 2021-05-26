import os
import glob
import argparse

import numpy as np

import place_blocks
from render_schematic import render

def create_html(im_dir, pattern='*180.png'):
    imgs = glob.glob(os.path.join(im_dir, pattern))
    with open(os.path.join(im_dir, 'all_label.html'), 'w') as f:
        for img_path in imgs:
            img_path = os.path.basename(img_path)
            f.write(f'<img src="{img_path}">{img_path}\n')

def remove_blocks(house, block_changes):
    xmax, ymax, zmax = house.shape[:-1]
    _, xb, _, zb = block_changes.max(0) - block_changes.min(0)
    block_changes = block_changes - [0, xb, 63, zb]
    for b, z, x, y in block_changes:
        if b == -1:
            house[x,y,z,:] = (0,0)
        else:
            if 0<x<xmax and 0<y<ymax and 0<z<zmax:
                # if y == -18: import pdb; pdb.set_trace()
                house[x,y,z,:] = (b,0)
            else:
                new_house = np.zeros(
                    (max(x, xmax), max(y, ymax), max(z, zmax), 2)
                )
                new_house[:xmax, :ymax, :zmax, :] = house
                new_house[x, y, z] = (b,0)
                house = new_house
    return house

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--npy_files',
        default='/craftassist/mturk_exp_houses.txt',
        help='file of npy file paths'
    )
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--pattern', default='*180.png')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--remove', default='./changes/3.23.103.34_2.npy')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.remove:
        block_changes = np.load(args.remove)
        house = np.load('/craftassist/minecraft_specs/schematics/cleaned_houses/validation3.npy')
        import pdb; pdb.set_trace()
        house_blocks = place_blocks.yzx_to_dicts(house)
        for b, x, y, z in block_changes:
            if b == -1:
                b = 0
            house_blocks.append({'x': x, 'y': y, 'z': z, 'id': b, 'meta': 0})

        # compute focus and distance for second house
        ymax, zmax, xmax, _ = house.shape
        ymid, zmid, xmid = ymax // 2, zmax // 2, xmax // 2
        focus = np.array([xmid, ymid + 63, zmid])  # TODO: +63 only works for flat_world seed=0
        distance = int((xmax ** 2 + zmax ** 2) ** 0.5)

        # render original house
        # render(
        #     '/craftassist/minecraft_specs/schematics/cleaned_houses/validation3.npy',
        #     args.out_dir,
        #     "flat_world",
        #     0,
        #     False,
        #     False,
        #     25565,
        #     None,
        #     None,
        #     100,
        #     [256, 256]
        # )

        # render modifications
        render(
            house_blocks,
            args.out_dir,
            "flat_world",
            0,
            False,
            False,
            25565,
            distance,
            None,
            100,
            [256, 256],
            focus=focus,
            npy_basename='tmp'
        )

    else:
        with open(args.npy_files, 'r') as f:
            to_vis = [l.strip() for l in f.readlines()]

        for fname in to_vis:
            render(
                fname,
                args.out_dir,
                "flat_world",
                0,
                False,
                False,
                25565,
                None,
                None,
                100,
                [256, 256]
            )
        create_html(args.out_dir, pattern=args.pattern)

