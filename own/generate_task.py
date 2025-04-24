

import argparse
import json
import torch
import shortuuid
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from own.preprocess_qwen import preprocess_qwen

import warnings
warnings.filterwarnings("ignore")


def generate_task_plan_from_images(model_path, model_base, image_paths, question_text, output_file):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name)

    print('Model is loaded successfully')

        # Prefix each image path with "./picture/"
    image_paths = [f"../pictures/{image_path}" for image_path in image_paths]

    image_tensors = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        image_tensors.append(image_tensor.half().cuda())

    # system message
    system_message = (
        "You are an AI system responsible for generating robot-executable plans.\n\n"
        "1. Observe the given images to understand the current situation.\n"
        "2. Based on the user message, define the high-level Task.\n"
        "3. Break it down into simple, clear Subtasks that a robot can follow.\n"
        "4. For each Subtask, identify possible exceptions or issues that might occur, and describe how to handle them.\n\n"
        "Structure your answer as follows:\n"
        "- Task: ...\n"
        "- Subtask: 1 - ..., 2 - ..., ...\n"
        "- Exception Cases: ...\n"
    )

    image_token_str = "<image>\n" * len(image_paths)
    prompt = [
        {
            "from": "human",
            "value": f"{image_token_str}{question_text}"
        },
        {
            "from": "gpt",
            "value": None
        }
    ]
    print('input will be preprocessed')

    input_ids = preprocess_qwen(prompt, tokenizer, has_image=True, system_message=system_message)
    input_ids = input_ids.cuda()

    print('input is preprocessed successfully')
    print("input_ids:", input_ids)
    print("type:", type(input_ids))
    print("image_tensors length:", len(image_tensors))

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensors,
            max_new_tokens=1024
        )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(output)
    exit()
    task = question_text.split('\n')[0].strip()
    subtasks = [l for l in lines if l[:1].isdigit()]
    exceptions = [l for l in lines if
                  any(keyword in l.lower() for keyword in ["exception", "issue", "problem", "note", "warning"])]

    result = {
        "Task": task,
        "Subtask": subtasks,
        "Exception Cases": exceptions
    }

    with open(output_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--images", nargs="+", required=True, help="List of image paths")
    parser.add_argument("--task", type=str, required=True, help="User instruction (question)")
    parser.add_argument("--output-file", type=str, default="task_output.json")
    args = parser.parse_args()

    generate_task_plan_from_images(
        model_path=args.model_path,
        model_base=args.model_base,
        image_paths=args.images,
        question_text=args.task,
        output_file=args.output_file
    )
