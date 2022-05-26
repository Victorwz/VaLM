# adapted from https://github.com/cybertronai/bflm/blob/master/eval_lambada.py

import torch
from fairseq.checkpoint_utils import load_model_ensemble
import clip
import sys
import argparse
import json
from torch.utils.data import DataLoader, Dataset

def argmax(t):
    return int(torch.argmax(t).item())

def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n'+text.strip()

def score_batch(args, model, enc, batch, device="cuda"):
    """Return number of last-word mismatches in a batch."""
    batch_encoded = []
    lengths = []
    fragments = []
    for line in batch:
        line = line.strip()
        if args.jeff_suggestion:
            line = '\n' + line
        line_encoded = enc.encode(line)
        line_encoded = list(map(lambda x: x + 4, line_encoded))
        # encoded_last_word = enc.decode(line_encoded[-1:]).strip()
        actual_last_word = line.split()[-1].strip()
        # if encoded_last_word != actual_last_word:
        if len(enc.encode(actual_last_word)) > 1:
            fragments.append(True)
        else:
            fragments.append(False)
        batch_encoded.append(line_encoded)

    # array is ragged, so pad to turn into rectangular tensor
    max_len = max(len(encoded) for encoded in batch_encoded)
    batch_padded = []
    for encoded in batch_encoded:
        batch_padded.append(encoded+ [1] * (max_len - len(encoded))) # pad is 1 for fairseq
        lengths.append(len(encoded))

    batch_padded = torch.tensor(batch_padded)
    batch_padded = batch_padded.to(device)
    if args.dryrun:
        return 0, 1
    
    logits, presents = model(batch_padded)
    logits = logits.softmax(dim=-1)

    errors = 0
    total = 0
    for i in range(args.batch):
        # break on small last batch
        if i >= len(batch_padded):
            break
        last_idx = lengths[i]-1
        observed = batch_encoded[i][last_idx]
        predicted = argmax(logits[i][last_idx-1])
        if args.ignore_fragments and fragments[i]:
            continue
        total += 1
        errors += 0 if (observed == predicted) else 1

    return errors, total

def main(args):
    ds_raw = open(f'{args.data_path}').read()
    if args.preprocess:
        ds_raw = preprocess(ds_raw)
        
    ds = ds_raw.strip().split('\n')

    # special handling for jsonl file
    lines = []
    if args.data_path.endswith('.jsonl'):
        # special handling for file from Jeff
        for line in ds:
            #            candidate1 = eval(line)['text']
            #            lines.append(candidate1)
            candidate2 = line[len('{"text": "'):-len('"}')]
            candidate2 = f'''"""{candidate2}"""'''
            lines.append(eval(candidate2))

            #            lines.append(eval(line))
            #print(line)
            #            break
            #            print(line)
            #            eprint(lines[-1])
        ds = lines

    data_loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    if args.model_overrides:
        # override_args = json.loads(overrides)
        override_args = {"dstore_filename": "./data/image_feature_datastore_200M", "use_gpu_to_search": False, } #  "use_knn_datastore": False, "load_knn_datastore": False}
    else:
        override_args = ""

    model, _ = load_model_ensemble([args.path], arg_overrides=override_args, task=None)
    model = model[0]
    mdoel = model.eval()
    model = model.cuda()

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    errors = 0
    total = 0
    for i, batch in enumerate(data_loader):
        errors_batch, total_batch = score_batch(args, model, tokenizer, batch)
        errors += errors_batch
        total += total_batch
        if args.max_batches and i >= args.max_batches-1:
            break

    print("Accuracy: %.4f"%(1 - errors / total,))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating VaLM")
    parser.add_argument("--path", type=str, default="/path/to/ckpt", help="The path to the model")
    parser.add_argument("--data-path", type=str, default="./data/lambada/lambada_test.jsonl", help="The path to the test data")
    parser.add_argument("--model-overrides", action="store_true", default=False, help="Overrides args for model")
    parser.add_argument("--batch", default=4, help="batch size")
    parser.add_argument("--ignore-fragments", action="store_true", help="Whether to run training.")
    parser.add_argument("--preprocess", action="store_true", help="strip quotes")
    parser.add_argument('--jeff_suggestion',  action='store_true', help="use jeff's suggestion of prepending \n to each example")
    parser.add_argument('--dryrun',  action='store_true', help="test preprocessing pipeline")
    parser.add_argument('--max-batches', type=int, default=0, help='batch size')
    args = parser.parse_args()
    main(args)

