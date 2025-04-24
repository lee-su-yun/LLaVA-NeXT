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
                → ["Turn on the light", "by touching the button", "."]
                → [tokens("Turn on the light"), IMAGE_TOKEN_INDEX, tokens("by touching..."), IMAGE_TOKEN_INDEX, tokens("."))]
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
    q_types = ["ego"] #["ego", "exo", "both", "any"]
    image_views = ["both"]

    for c in category:
        if c != args.category and args.category!="total":
            continue
        for q_type in  q_types:
            question_file = args.question_file + c + "_" + q_type + ".jsonl"
            # Data
            questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
            questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
            _answers_file = args.answers_file + "llava-next-interleave-7b" + "/" + c + '_' + q_type + "(both).jsonl"
            answers_file = os.path.expanduser(_answers_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            ans_file = open(answers_file, "w")

            system_message = "You are provided with two visual inputs in sequence, each captured from a different perspective:\n1. The view from the camera worn by the user ('I').\n2. The view captured by an external camera observing the user ('I').\nThese two images capture the same event at the same time.\nYour task is to analyze both images along with the question and provide the most accurate response based on the visual information from both perspectives.\n"
            count = 0
            for line in tqdm(questions):
                if count == 5:
                    break
                count +=1
                idx = line["question_id"]
                base = line["base"]
                image_file_ego = line["image_ego"]
                image_file_exo = line["image_exo"]

                image_tensors = []
                image_ego = Image.open(os.path.join(args.image_folder, base, image_file_ego))
                image_tensor_ego = image_processor.preprocess(image_ego, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor_ego.half().cuda())
                image_exo = Image.open(os.path.join(args.image_folder, base, image_file_exo))
                image_tensor_exo = image_processor.preprocess(image_exo, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor_exo.half().cuda())

                qs = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n' + line['text']

                cot_qs = [
                    (f"{qs}\n\nLet's think step-by-step as follows:\n"
                    "1. Consider only the first image. Based on the first image alone, how would you answer the question? Write down the reasoning and tentative answer.\n"),
                    "2. Consider only the second image. Based on the second image alone, how would you answer the question? Write down the reasoning and tentative answer.\n",
                    "3. Compare the observations from both images. Identify any overlapping or identical objects, features, or contextual clues that could link the two images. Integrate these findings to develop a unified perspective.\n",
                    "4. Based on the integrated analysis, provide the final answer to the original question using only 'yes' or 'no'."
                ]


                outputs = []
                input_ids = None
                for i in range(len(cot_qs)):
                    qs = cot_qs[i]
                    prompt = []
                    prompt_h = {
                        "from": "human",
                        "value": qs,
                    }
                    prompt.append(prompt_h)
                    prompt_g = {
                        'from': 'gpt',
                        'value': None
                    }
                    prompt.append(prompt_g)


                    if input_ids is None:
                        input_ids = preprocess_qwen(prompt, tokenizer, has_image=True,system_message=system_message).cuda()
                    else:
                        input_ids_new = preprocess_qwen(prompt, tokenizer, has_image=True, system_message=system_message).cuda()
                        input_ids = torch.cat((input_ids, output_ids ,input_ids_new), dim=1)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensors,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            # no_repeat_ngram_size=3,
                            max_new_tokens=1024,
                            use_cache=True)

                    outputs.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({"question_id": idx,
                                           "text_0":outputs[0],
                                           "text_1": outputs[1],
                                           "text_2": outputs[2],
                                           "text_3": outputs[3],
                                           "answer_id": ans_id,
                                           "model_id": model_name}) + "\n")
            ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-next-interleave-qwen-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/sdc1/datasets/EEH_v2/test")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--category", type=str, choices=["obj", "self", "spatial", "total"], default="obj")
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)
