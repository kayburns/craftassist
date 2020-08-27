# flake8: noqa

import json
import math
import pickle
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

from utils_caip import *
from utils_parsing import *
from train_model import *

from pprint import pprint

model = "python/craftassist/models/ttad_bert_updated/model/caip_test_model.pth"
args_path = "python/craftassist/models/ttad_bert_updated/model/caip_test_model_args.pk"
args = pickle.load(open(args_path, "rb"))

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
full_tree, tree_i2w = json.load(open(args.tree_voc_file))
dataset = CAIPDataset(tokenizer, args, prefix="", full_tree_voc=(full_tree, tree_i2w))

enc_model = AutoModel.from_pretrained(args.pretrained_encoder_name)
bert_config = BertConfig.from_pretrained("bert-base-uncased")
bert_config.is_decoder = True
bert_config.vocab_size = len(tree_i2w) + 8
bert_config.num_hidden_layers = args.num_decoder_layers
dec_with_loss = DecoderWithLoss(bert_config, args, tokenizer)
encoder_decoder = EncoderDecoderWithLoss(enc_model, dec_with_loss, args)
encoder_decoder.load_state_dict(torch.load(model))

encoder_decoder = encoder_decoder.cuda()
_ = encoder_decoder.eval()


def get_beam_tree(chat, noop_thres=0.95, beam_size=5, well_formed_pen=1e2):
    btr = beam_search(chat, encoder_decoder, tokenizer, dataset, beam_size, well_formed_pen)
    if btr[0][0].get("dialogue_type", "NONE") == "NOOP" and math.exp(btr[0][1]) < noop_thres:
        tree = btr[1][0]
    else:
        tree = btr[0][0]
    return tree
