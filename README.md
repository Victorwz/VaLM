# VaLM
Official implementation of our paper "[Visually-Augmented Language Modeling](https://arxiv.org/abs/2205.10178)". Please cite our paper if you find this repository helpful in your research:
```
@article{Wang2022VisuallyAugmentedLM,
  title={Visually-Augmented Language Modeling},
  author={Weizhi Wang and Li Dong and Hao Cheng and Haoyu Song and Xiaodong Liu and Xifeng Yan and Jianfeng Gao and Furu Wei},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.10178}
}
```

## Environment Setup 
Create a virtual environment and run 
```
bash setup.sh
```
Then the revised `fairseq` and ohter packages will be installed. We strongly recommend you to use python version >=3.6 <=3.8 for stability.

## Text and Image Data Preparation
* Preprocessing text training data:
```
bash myscripts/preprocess_valm_text.sh
```
The cc100 English original corpus would be available at [CC100-EN](https://data.statmt.org/cc-100/en.txt.xz). The sharding script is available at `./data/roberta-cc100-ori/sharded_data.py`.

* Preprocessing image data:

Please refer to [LAION](https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/) for downloading the image dataset for creating image visual knowledge base.
```
- portion=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
- python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features_raw \
    --image_data_path /multimodal/data/image/laion_all \
    --tar_id_start 0 --tar_id_end 20000 \
    --n_gpus 16 \
    --portion {portion}
```

Once we get the image features, we could make a sanity check by
```
python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features_raw \
    --image_data_path /multimodal/data/image/laion_all \
    --verify
```
where it will retrieve from the first five shards, and the output should be:
```
    imageRetriever.retrieve("A cute cat")  # 000048808.jpg √
    imageRetriever.retrieve("A cute dog")  # 000032573.jpg √
```

Path to the processed image features:
```
/mnt/multimodal/VaLM/image_features_raw
```
Each file's name is like `img_features_12345.pt`, where `12345` is the id of the laion tar file.


## Visual Knowledge Base Creation and Text-to-Image Retrieval
* Constructing cached datastore of image features:
```
DSTORE_PATH=./data/image_feature_datastore_200M

python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features_raw \
    --image_data_path /multimodal/data/image/laion_all \
    --save_image_datastore --dstore_mmap $DSTORE_PATH --dstore_fp16 \
    --dstore_size 191504487
```

* Training faiss index of cached datastore:
```
DSTORE_PATH=./data/image_feature_datastore_200M

python ImageRetrieval/train_datastore_gpu.py --dstore_size 191504486 \
    --dstore_mmap $DSTORE_PATH \
    --dstore_fp16 --dimension 768 --ncentroids 131072
```

* Verify retrieval with samples:
```
DSTORE_PATH=./data/image_feature_datastore_200M

python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features \
    --image_data_path /multimodal/data/image/laion_all \
    --verify_retriever --dstore_mmap $DSTORE_PATH \
    --dstore_filename $DSTORE_PATH --dstore_fp16 \
    --dstore_size 191504486
```
* The demo retrieval results will be write to `./html/reports.html`. Download the `html` folder to see the results. 

## Training GPT-See
* Example training command on multiple data shards with 16 Tesla-V100 gpus:
```
bash myscripts/train_valm.sh
```

For training text-only baseline GPT-BLIND, run:
```
bash myscripts/train_gpt_blind.sh
```

## Evaluating VaLM
* Evaluate the trained checkpoint on object color reasoning:
```
python evaluation_scripts/verify_color_prediction.py --path /path/to/ckpt --model-overrides
```

* Evaluate the trained checkpoint on object size reasoning:
```
python evaluation_scripts/verify_size_reason.py --path /path/to/ckpt --model-overrides
```

* Evaluate the trained checkpoint on language modeling:
```
fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --path /path/to/ckpt
fairseq-eval-lm ./data/lambada/ --batch-size 4 --sample-break-mode eos --path /path/to/ckpt
python evaluation_scripts/eval_lambada.py --data-path ./data/lambada/lambada_test.jsonl --preprocess --path /path/to/ckpt
```
The script for selecting best checkpoint on validation set is available at `./evaluation_scripts/ckpt_selection_valid.py`.

<!-- # Model Architectures
| ARCH               | emb\_dim | ffn\_dim | layers | att\_heads | dropout | att\_dropout | act\_fn |
|-----------------------|---------|---------|--------|-----------|---------|-------------|--------|
| valm\_gpt          | 768     | 3072    | 12     | 12        | 0.1     | 0.1         | gelu   |
| valm\_gpt2\_small  | 1024    | 4096    | 24     | 16        | 0.1     | 0.1         | gelu   |
| valm\_gpt2\_tiny   | 64      | 64      | 2      | 1         | 0.1     | 0.1         | gelu   |
| valm\_gpt2\_medium | 1280    | 5120    | 36     | 20        | 0.1     | 0.1         | gelu   |
| valm\_gpt2\_big    | 1600    | 6400    | 48     | 25        | 0.1     | 0.1         | gelu   |
| valm\_gpt3\_small  | 768     | 3072    | 12     | 12        | 0       | 0           | gelu   |
| valm\_gpt3\_medium | 1536    | 6144    | 24     | 16        | 0       | 0           | gelu   |
| valm\_gpt3\_large  | 2048    | 8192    | 24     | 32        | 0       | 0           | gelu   |
| valm\_gpt3\_xl     | 2560    | 10240   | 32     | 32        | 0       | 0           | gelu   |
| valm\_gpt3\_6\_7   | 4096    | 16384   | 32     | 32        | 0       | 0           | gelu   |
| valm\_gpt3\_13     | 5120    | 20480   | 40     | 40        | 0       | 0           | gelu   |
| valm\_gpt3\_175    | 12288   | 49152   | 96     | 96        | 0       | 0           | gelu   | -->

## Future Work
We are currently working on the following directions to push VaLM to a higher level:
* Adapt VaLM to vision-language tasks, especially image captioning and visual question-answering
* Train larger size VaLM, i.e. VaLM-Medium, VaLM-Large, VaLM-XL
* Adapt VaLM to a Encoder-Decoder architecture for NLG tasks

If you are interested in cooperation or have fantastic ideas based on VaLM, please contact weizhiwang AT ucsb.edu or leave some Git issues.

## Credits
VaLM is developed based on [fairseq](https://github.com/facebookresearch/fairseq) and [CLIP](https://github.com/openai/CLIP).