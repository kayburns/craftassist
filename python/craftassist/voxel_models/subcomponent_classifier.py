"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import queue
from multiprocessing import Queue, Process
import sys
import os
from mc_memory_nodes import InstSegNode, PropSegNode
from heuristic_perception import all_nearby_objects
from shapes import get_bounds

VISION_DIR = os.path.dirname(os.path.realpath(__file__))
CRAFTASSIST_DIR = os.path.join(VISION_DIR, "../")
SEMSEG_DIR = os.path.join(VISION_DIR, "semantic_segmentation/")
sys.path.append(CRAFTASSIST_DIR)
sys.path.append(SEMSEG_DIR)

import build_utils as bu
from semseg_models import SemSegWrapper


# TODO all "subcomponent" operations are replaced with InstSeg
class SubcomponentClassifierWrapper:
    def __init__(self, agent, model_path, vocab_path, perceive_freq=0):
        self.agent = agent
        self.memory = self.agent.memory
        self.perceive_freq = perceive_freq
        self.true_temp = 1
        if model_path is not None:
            self.subcomponent_classifier = SubComponentClassifier(
                voxel_model_path=model_path, vocab_path=vocab_path,
            )
            self.subcomponent_classifier.start()
        else:
            self.subcomponent_classifier = None

    def perceive(self, force=False):
        if self.perceive_freq == 0 and not force:
            return
        if self.perceive_freq > 0 and self.agent.count % self.perceive_freq != 0 and not force:
            return
        if self.subcomponent_classifier is None:
            return
        # TODO don't all_nearby_objects again, search in memory instead
        to_label = []
        # add all blocks in marked areas
        for pos, radius in self.agent.areas_to_perceive:
            for obj in all_nearby_objects(self.agent.get_blocks, pos, radius):
                to_label.append(obj)
        # add all blocks near the agent
        for obj in all_nearby_objects(self.agent.get_blocks, self.agent.pos):
            to_label.append(obj)

        for obj in to_label:                                                    # (6, 69, 11) in [b[0] for b in obj]
            self.subcomponent_classifier.block_objs_q.put(obj)

        # everytime we try to retrieve as many recognition results as possible
        while not self.subcomponent_classifier.loc2labels_q.empty():
            loc2labels, obj = self.subcomponent_classifier.loc2labels_q.get()   # (6, 69, 11) in [b[0] for b in obj]

            loc2ids = dict(obj)
            label2blocks = {}

            def contaminated(blocks):
                """
                Check if blocks are still consistent with the current world
                """
                mx, Mx, my, My, mz, Mz = get_bounds(blocks)
                yzxb = self.agent.get_blocks(mx, Mx, my, My, mz, Mz)
                for b, _ in blocks:
                    x, y, z = b
                    if loc2ids[b][0] != yzxb[y - my, z - mz, x - mx, 0]:
                        return True
                return False

            for loc, labels in loc2labels.items():
                b = (loc, loc2ids[loc])
                for l in labels:
                    if l in label2blocks:
                        label2blocks[l].append(b)
                    else:
                        label2blocks[l] = [b]
            
            labels_str = " ".join(list(label2blocks.keys()))
            if len(labels_str) == 1:
                self.agent.send_chat(
                    "I found this in the scene: " + labels_str
                )
            elif len(labels_str) > 1:
                self.agent.send_chat(
                    "I found these in the scene: " + labels_str
                )
            for l, blocks in label2blocks.items():
                ## if the blocks are contaminated we just ignore
                if not contaminated(blocks):
                    #locs = [loc for loc, idm in blocks]
                    InstSegNode.create(
                        self.memory, blocks, [l, 'semseg'])

    def update(self, label, blocks, house):
        pass
        #self.subcomponent_classifier.to_update_q.put((label, blocks, house))


class SubComponentClassifier(Process):
    """
    A classifier class that calls a voxel model to output object tags.
    """

    def __init__(self, voxel_model_path=None, vocab_path=None, true_temp=1):
        super().__init__()

        if voxel_model_path is not None:
            logging.info(
                "SubComponentClassifier using voxel_model_path={}".format(voxel_model_path)
            )
            self.model = SemSegWrapper(voxel_model_path, vocab_path)
        else:
            raise Exception("specify a segmentation model")

        self.block_objs_q = Queue()  # store block objects to be recognized
        self.loc2labels_q = Queue()  # store loc2labels dicts to be retrieved by the agent
        #self.to_update_q = Queue()
        self.daemon = True

    def run(self):
        """
        The main recognition loop of the classifier
        """
        while True:  # run forever
            #for _ in range(100):
            #    print("If I print here, it solves the bug ¯\_(ツ)_/¯, priority thing?")
            tb = self.block_objs_q.get(block=True, timeout=None)
            loc2labels = self._watch_single_object(tb)
            for k in loc2labels.keys():
                loc2labels[k].append("house")

            self.loc2labels_q.put((loc2labels, tb))
            #try:
            #    label, blocks, house = self.to_update_q.get_nowait()
            #    self.update(label, blocks, house)
            #except queue.Empty:
            #    pass

    def _watch_single_object(self, tuple_blocks, t=1):
        """
        Input: a list of tuples, where each tuple is ((x, y, z), [bid, mid]). This list
               represents a block object.
        Output: a dict of (loc, [tag1, tag2, ..]) pairs for all non-air blocks.
        """

        def get_tags(p):
            """
            convert a list of tag indices to a list of tags
            """
            return [self.model.tags[i][0] for i in p]

        def apply_offsets(cube_loc, offsets):
            """
            Convert the cube location back to world location
            """
            return (cube_loc[0] + offsets[0], cube_loc[1] + offsets[1], cube_loc[2] + offsets[2])

        np_blocks, offsets = bu.blocks_list_to_npy(blocks=tuple_blocks, xyz=True)

        pred = self.model.segment_object(np_blocks, T=t)

        # convert prediction results to string tags
        return dict([(apply_offsets(loc, offsets), get_tags([p])) for loc, p in pred.items()])

    def recognize(self, list_of_tuple_blocks):
        """
        Multiple calls to _watch_single_object
        """
        tags = dict()
        for tb in list_of_tuple_blocks:
            tags.update(self._watch_single_object(tb))
        return tags

    def update(self, label, blocks, house):
        # changes can come in from adds or removals, if add, update house
        logging.info("Updated label {}".format(label))
        if blocks[0][0][0] > 0:
            house += blocks 
        blocks = [(xyz, (1, 0)) for xyz, _ in blocks]
        np_house, offsets = bu.blocks_list_to_npy(blocks=house, xyz=True)
        np_blocks, _ =  bu.blocks_list_to_npy(
            blocks=blocks, xyz=False, offsets=offsets, shape=np_house.shape) # shape is still xyz bc of shape arg
        self.model.update(label, np_blocks, np_house)

