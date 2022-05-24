DSTORE_PATH=./data/image_feature_datastore_200M

python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features_raw \
    --image_data_path /multimodal/data/image/laion_all \
    --save_image_datastore --dstore_mmap $DSTORE_PATH --dstore_fp16 \
    --dstore_size 191504487 # a small bug that this number should be #imgs+1; would fix later

python ImageRetrieval/train_datastore_gpu.py --dstore_size 191504486 \
    --dstore_mmap $DSTORE_PATH \
    --dstore_fp16 --dimension 768 --ncentroids 131072

python ImageRetrieval/clip_image_retrieval.py --mount /mnt --ifp /multimodal/VaLM/image_features \
    --image_data_path /multimodal/data/image/laion_all \
    --verify_retriever --dstore_mmap $DSTORE_PATH \
    --dstore_filename $DSTORE_PATH --dstore_fp16 \
    --dstore_size 191504486