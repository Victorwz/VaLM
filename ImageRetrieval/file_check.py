import sys
import json
import os
import numpy as np
import logging
from argparse import ArgumentParser

parser = ArgumentParser("extracted feature statistics")
parser.add_argument("--mount", type=str, default="./data")
parser.add_argument("--ifp", type=str, default="/multimodal/VaLM/image_features")
parser.add_argument("--image_data_path", type=str, default="/multimodal/data/image/laion_all")

args = parser.parse_args()

N = 20000

if __name__ == '__main__':
    f_path = f"{args.mount}{args.ifp}"
    extracted_ids = []
    f_names = os.listdir(f_path)
    for n in f_names:
        num = n.replace(".pt","").split("_")[-1]
        extracted_ids.append(int(num))

    missing_ids = []
    for i in range(N):
        if i not in extracted_ids:
            missing_ids.append(i)
    print(missing_ids)
    # print(extracted_ids)
    print(len(missing_ids))
