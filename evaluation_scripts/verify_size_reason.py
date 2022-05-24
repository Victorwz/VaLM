import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json

prompt_list = [
    "Is [ITEMA] bigger than [ITEMB]?",
    "Is [ITEMA] larger than [ITEMB]?",
    "Is [ITEMA] taller than [ITEMB]?",
    "[ITEMA] is bigger than [ITEMB], is it true?"
    "[ITEMA] is larger than [ITEMB], is it true?"
    "[ITEMA] is taller than [ITEMB], is it true?"
]

def main(model_path="", data_path="", overrides=""):
    if overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/image_feature_datastore_200M", "use_gpu_to_search": False} 
        # remove the key-value pair "use_gpu_to_search": False from overrides_args dict if your gpu memory is larger than 20G
        # add the key-value pairs "use_knn_datastore": False, "load_knn_datastore": False to the overrides_arg dict for evaluating ablation baseline VaLM-distillation
    else:
        override_args = ""
    
    model, _ = load_model_ensemble([model_path], arg_overrides=override_args, task=None)
    model = model[0]
    model = model.cuda()
    model = model.eval()
    
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    larger_index = tokenizer.encode("Yes")[0] + 4
    smaller_index = tokenizer.encode("No")[0] + 4
    
    total_cnt = 0
    acc_cnt = 0 
    for prompt in prompt_list:
        with open(data_path) as f:
            for idx, line in enumerate(f.readlines()):
                bigger_item, smaller_item = line.strip("\n").split()
                with torch.no_grad():
                    total_cnt += 1
                    query = prompt.replace("[ITEMA]", bigger_item).replace("[ITEMB]", smaller_item)

                    tokens = torch.tensor(tokenizer.encode(query)) + 4
                    tokens = tokens.unsqueeze(0).cuda()
                    prediction = model(tokens, features_only=False)
                    prediction = prediction[0][0, -1, :].softmax(dim=-1).cpu()

                    if prediction[larger_index] > prediction[smaller_index]:
                        acc_cnt += 1

    print("Object Size Reasoning Accuracy is : {}".format(acc_cnt / total_cnt))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating GPT-See")
    parser.add_argument("--path", type=str, default="/path/to/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./object_size/sizePairsFull.txt", help="The path to the data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    args = parser.parse_args()
    main(args.path, data_path=args.data_path, overrides=args.model_overrides)