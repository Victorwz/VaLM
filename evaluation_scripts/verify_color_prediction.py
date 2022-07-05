import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json

prompt_list = [
    "Q: What is the color of [DESCRIPTOR] [ITEM]? A: It is",
    "What is the color of [DESCRIPTOR] [ITEM]? It is",
    "What is the usual color of [DESCRIPTOR] [ITEM]?",
    "What is the typical color of [DESCRIPTOR] [ITEM]?",
    "The color of [DESCRIPTOR] [ITEM] is",
    "The usual color of [DESCRIPTOR] [ITEM] is",
    "The common color of [DESCRIPTOR] [ITEM] is",
    "The typical color of [DESCRIPTOR] [ITEM] is",
    "[DESCRIPTOR] [ITEM] usually has the color of",
]

def main(model_path="", data_path="", overrides=False):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/image_feature_datastore_200M", "use_gpu_to_search": False} 
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""

    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.eval()
    model = model.cuda()

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    color_set = set()
    with open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue
            line = line.strip("\n").split(",")
            if len(line) == 2:
                concrete_object, color = line[-2:]
            elif len(line) == 4:
                descriptor, concrete_object, color = line[-3:]
            color_token = tokenizer.encode(color)
            assert len(color_token) == 1
            color_set.add(color_token[0])

    total_cnt = 0
    acc_cnt = 0
    for prompt in prompt_list:
        with open(data_path) as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue # skip title line
                total_cnt += 1
                line = line.strip("\n").split(",")
                if len(line) == 2:
                    concrete_object, color_ground_truth = line[-2:]
                    query = prompt.replace("[ITEM]", concrete_object.strip())
                elif len(line) == 4:
                    descriptor, concrete_object, color_ground_truth = line[-3:]
                    if descriptor:
                        query = prompt.replace("[ITEM]", concrete_object.strip()).replace("[DESCRIPTOR]", descriptor.strip())
                    else:
                        query = prompt.replace("[ITEM]", concrete_object.strip()).replace("[DESCRIPTOR] ", "")
                    # print(query)
                with torch.no_grad():
                    tokens = tokenizer.encode(query)
                    tokens = torch.tensor(tokens) + 4 # fairseq dict would append eos bos pad unk to the beginning of vocab, leading to a positional difference of 4
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = prediction[0][0, -1, :].softmax(dim=-1).cpu()
                    predicted_color = {tokenizer.decode([color_index]): prediction[color_index+4].item() for color_index in color_set}
                    predicted_color = sorted(predicted_color.items(), key=lambda x: x[1], reverse=True)
                    # print(predicted_color[0][0].strip(), color_ground_truth.strip())
                    if predicted_color[0][0].strip() == color_ground_truth.strip():
                        acc_cnt += 1

    print("Object Color Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/VaLM/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/object_color/memory_color_data.csv", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--color-terms", action="store_true", default=False, help="Using Color-Terms dataset")
    args = parser.parse_args()
    if args.color_terms:
        args.data_path = "./data/object_color/color-concrete-objects.csv"
    main(args.path, args.data_path, overrides=args.model_overrides)
