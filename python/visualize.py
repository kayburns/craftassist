import os
import glob
import argparse

from filter_validation_homes import filter_validation
from render_schematic import render

def create_html(im_dir, pattern='*180.png'):
    imgs = glob.glob(os.path.join(im_dir, pattern))
    with open(os.path.join(im_dir, 'all_label.html'), 'w') as f:
        for img_path in imgs:
            img_path = os.path.basename(img_path)
            f.write(f'<img src="{img_path}">{img_path}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument(
        '--filter', action='store_true',
        help='whether or not to filter npy files for good form (used in ' \
            'creating good validation set'
    )
    parser.add_argument('--pattern', default='*180.png')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.filter:
        to_vis = filter_validation(args.npy_dir)
    else:
        to_vis = glob.glob(os.path.join(args.npy_dir, '*.npy'))

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

