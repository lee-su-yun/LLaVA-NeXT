
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import math
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
   # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates, target is the answer(not for inference)
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], [] # For one chat (single turn)
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens # system message tokenize
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens # _system + tokenizer(system_message).input_ids : IGNORE_INDEX
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source): # source = {"from": "human", "value": "~~~~"}, {"from": "gpt", "value": "~~~~"}, ...
        role = roles[sentence["from"]] # human -> <|im_start|>user, gpt -> <|im_start|>assistant
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>') # split based on image
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens # [text1+image1+text2+image2+text3] (prompt) -> [im_start]+[text1]+[image1]+[text2]+[image2]+[text3]+[im_end] (token)
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
            """
            "Turn on the light <image> by touching the button <image>."
                ?~F~R ["Turn on the light", "by touching the button", "."]
                ?~F~R [tokens("Turn on the light"), IMAGE_TOKEN_INDEX, tokens("by touching..."), IMAGE_TOKEN_INDEX, tokens("."))]
            """
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            """
            prompt : "Where is the cat?" <image>
            _input_id = [<|im_start|>user, \n, tokens("Where is the cat?"), tokens(<image>), <|im_end|>, \n]
            """
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens # If target exist, change the target
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    category = ["obj", "self", "spatial"]
    q_types = ["ego"]  # ["ego", "exo", "both", "any"]
    image_views = ["both"]

    for c in category:
        if c != args.category and args.category != "total":
            continue
        for q_type in q_types:
            question_file = args.question_file + c + "_" + q_type + ".jsonl"
            # Data
            questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
            questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
            _answers_file = args.answers_file + "llava-next-interleave-7b" + "/" + c + '_' + q_type + "(both).jsonl"
            answers_file = os.path.expanduser(_answers_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            ans_file = open(answers_file, "w")

            system_message = "You are provided with two visual inputs in sequence, each captured from a different perspective:\n1. The view from the camera worn by the user ('I').\n2. The view captured by an external camera observing the user ('I').\nThese two images capture the same event at the same time.\nYour task is to analyze both images along with the question and provide the most accurate response based on the visual information from both perspectives.\n"
