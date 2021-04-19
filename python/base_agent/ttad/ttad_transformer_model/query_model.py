import json
import math
import pickle
import logging

import torch

from transformers import AutoModel, AutoTokenizer, BertConfig

from utils_parsing import *
from utils_caip import *
from train_model import *


class TTADBertModel(object):
    def __init__(self, model_dir, data_dir, model_name="caip_test_model", decomposition_model=None):
        model_name = model_dir + model_name
        args = pickle.load(open(model_name + "_args.pk", "rb"))

        args.data_dir = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
        full_tree, tree_i2w = json.load(open(model_name + "_tree.json"))
        self.dataset = CAIPDataset(
            self.tokenizer, args, prefix="", full_tree_voc=(full_tree, tree_i2w)
        )

        enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_config.is_decoder = True
        bert_config.vocab_size = len(tree_i2w) + 8

        bert_config.num_hidden_layers = args.num_decoder_layers
        dec_with_loss = DecoderWithLoss(bert_config, args, self.tokenizer)
        self.encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.encoder_decoder.load_state_dict(
            torch.load(model_name + ".pth", map_location=map_location), strict=False
        )
        self.encoder_decoder = (
            self.encoder_decoder.cuda()
            if torch.cuda.is_available()
            else self.encoder_decoder.cpu()
        )
        self.encoder_decoder.eval()
        
        if decomposition_model:
            self.decomposition_model = decomposition_model()
        else:
            self.decomposition_model = decomposition_model

    def get_reps(self, chat):
        x_reps, batch, idx_rev_map, model_device = get_reps(
            chat, self.encoder_decoder, self.tokenizer, self.dataset
        )
        return x_reps

    def parse(self, chat, noop_thres=0.95, beam_size=5, well_formed_pen=1e2):
        if self.decomposition_model:
            chats = [chat]
            # continue decomposing until all decomposable chats are decomposed
            while any(isinstance(chat, str) for chat in chats):
                decomposed_chats = []
                for chat in chats:
                    if isinstance(chat, str):
                        x_reps, batch, idx_rev_map, model_device = get_reps(
                            chat, self.encoder_decoder, self.tokenizer,
                            self.dataset
                        )
                        decomp_res = self.decomposition_model.maybe_get_parse(x_reps)
                        if decomp_res:
                            logging.info("Decomposing {}".format(chat))
                            decomposed_chats.extend(decomp_res)
                        else:
                            decomposed_chats.append(
                                (chat, x_reps, idx_rev_map, batch)
                            )
                    else:
                        decomposed_chats.append(chat)
                chats = decomposed_chats

            # retrieve dictionary for each chat
            commands = []
            for chat, x_reps, idx_rev_map, batch in chats:
                btr = beam_search(
                    x_reps, batch, idx_rev_map, self.encoder_decoder, model_device,
                    self.dataset, beam_size, well_formed_pen 
                )
                if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
                    command = btr[1][0]
                else:
                    command = btr[0][0]
                commands.append((chat, command))

            # reduce down to one dictionary by linking action_sequence lists
            tree = commands[0][1]
            if len(commands) > 1 and tree.get("dialogue_type", "NONE") == "HUMAN_GIVE_COMMAND":
                for command in commands[1:]:
                    tree['action_sequence'].extend(command[1]['action_sequence'])
            return [c[0] for c in commands], tree

        else:
            x_reps, batch, idx_rev_map, model_device = get_reps(
                chat, self.encoder_decoder, self.tokenizer, self.dataset
            )
            btr = beam_search(
                x_reps, batch, idx_rev_map, self.encoder_decoder, model_device,
                self.dataset, beam_size, well_formed_pen 
            )
            if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
                tree = btr[1][0]
            else:
                tree = btr[0][0]
        return tree
