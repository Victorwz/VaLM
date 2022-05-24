for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    python myscripts/clip_tokenizer.py ./data/roberta-cc100-ori/train.${i}.txt ./data/VaLM_pretrain/bpe.${i}
    
    fairseq-preprocess --only-source --srcdict ./data/clip.vocab \
        --trainpref ./data/VaLM_pretrain/bpe.$i \
        --dest-dir ./data/data-bin/VaLM_pretrain/$i --workers 20
done # The multiprocess tool like FIFO for shell could be adopted to speed up the tokenization

