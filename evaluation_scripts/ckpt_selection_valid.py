import os, sys

model_path = sys.argv[1]
task = sys.argv[2]

for ckpt in os.listdir(model_path):
    if len(ckpt.split("_")) > 2 and int(ckpt.split("_")[1]) > 9:
        print("The model for evaluation is %s" % ckpt)
        if task == "s3vqa":
            if "gpt_see" in model_path:
                print("Normal evaluation without ground truth images")
                return_val = os.system("python evaluation_scripts/eval_s3vqa.py --path {}".format(os.path.join(model_path, ckpt)))
                print("Evaluation with ground truth images")
                return_val = os.system("python evaluation_scripts/eval_s3vqa.py --path {} --gt-image".format(os.path.join(model_path, ckpt)))
            else:
                return_val = os.system("python evaluation_scripts/eval_s3vqa.py --path {}".format(os.path.join(model_path, ckpt)))
        elif task == "lambada":
            return_val = os.system("python evaluation_scripts/eval_lambada.py --data-path ./data/lambada/lambada_development_plain_text.txt --preprocess --path {}".format(os.path.join(model_path, ckpt)))
            return_val = os.system("fairseq-eval-lm ./data/lambada/ --batch-size 4 --sample-break-mode eos --path {} --gen-subset valid".format(os.path.join(model_path, ckpt)))
        elif task == "wikitext":
            return_val = os.system("fairseq-eval-lm ./data/wikitext-103/ --batch-size 4 --sample-break-mode eos --path {} --gen-subset valid".format(os.path.join(model_path, ckpt)))
