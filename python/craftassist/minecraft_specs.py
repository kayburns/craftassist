"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import pickle

PATH = os.path.join(os.path.dirname(__file__), "../../minecraft_specs")
assert os.path.isdir(PATH), (
    "Not found: "
    + PATH
    + "\n\nDid you follow the instructions at "
    + "https://github.com/fairinternal/minecraft#getting-started"
)


_BLOCK_DATA = None


def get_block_data():
    global _BLOCK_DATA
    if _BLOCK_DATA is None:
        _BLOCK_DATA = _pickle_load("block_images/block_data")
    return _BLOCK_DATA


_COLOUR_DATA = None


def get_colour_data():
    global _COLOUR_DATA
    if _COLOUR_DATA is None:
        _COLOUR_DATA = _pickle_load("block_images/color_data")
    return _COLOUR_DATA


_BLOCK_PROPERTY_DATA = None


def get_block_property_data():
    global _BLOCK_PROPERTY_DATA
    if _BLOCK_PROPERTY_DATA is None:
        _BLOCK_PROPERTY_DATA = _pickle_load("block_images/block_property_data")
    return _BLOCK_PROPERTY_DATA


_MOB_PROPERTY_DATA = None


def get_mob_property_data():
    global _MOB_PROPERTY_DATA
    if _MOB_PROPERTY_DATA is None:
        _MOB_PROPERTY_DATA = _pickle_load("block_images/mob_property_data")
    return _MOB_PROPERTY_DATA


def get_bid_to_colours():
    bid_to_name = get_block_data()["bid_to_name"]
    name_to_colors = get_colour_data()["name_to_colors"]
    bid_to_colors = {}
    for item in bid_to_name.keys():
        name = bid_to_name[item]
        if name in name_to_colors:
            color = name_to_colors[name]
            bid_to_colors[item] = color
    return bid_to_colors

def _pickle_load(relpath):
    with open(os.path.join(PATH, relpath), "rb") as f:
        return pickle.load(f)
