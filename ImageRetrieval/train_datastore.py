import argparse
import os
import numpy as np
import faiss
import time
import logging
import sys

def train(dstore_mmap=None, dstore_size=None, dstore_fp16=True):

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger("clip.image.retrieval.train_datastore")

    # the implementation refers to knnlm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore_mmap', type=str, default='/mnt/multimodal/image_feature_datastore_200M', help='memmap where keys and vals are stored')
    parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    parser.add_argument('--dimension', type=int, default=768, help='Size of each key')
    parser.add_argument('--dstore_fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for sampling the subset of vectors to train the cache')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
    parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
    parser.add_argument('--num_keys_to_add_at_a_time', default=100000, type=int,
                        help='can only load a certain amount of data to memory at a time.')
    parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')
    parser.add_argument('--load_multiple_files', default=False, action='store_true')
    parser.add_argument('--multiple_key_files', type=str, default=None)
    parser.add_argument('--multiple_val_files', type=str, default=None)
    parser.add_argument('--multiple_files_size', type=str, default=None)
    parser.add_argument('--concat_file_path', type=str, default=None)

    args = parser.parse_args()

    # modify the args according to saved datastore
    if dstore_mmap:
        args.dstore_mmap = dstore_mmap
    if dstore_size:
        args.dstore_size = dstore_size
    args.faiss_index = args.dstore_mmap + "/knn_index"
    args.dstore_fp16 = dstore_fp16

    logger.info(args)

    res = faiss.StandardGpuResources()

    # load the saved keys and values
    if args.dstore_fp16:
        if args.load_multiple_files:
            assert args.multiple_key_files is not None and args.multiple_val_files is not None
            key_files = args.multiple_key_files.split(':')
            val_files = args.multiple_val_files.split(':')
            sizes = [int(size) for size in args.multiple_files_size.split(':')]
            logger.info(str(sizes))
            key_list = [np.memmap(key_file, dtype=np.float16, mode='r', shape=(sizes[idx], args.dimension)) for
                        idx, key_file in enumerate(key_files)]
            val_list = [np.memmap(val_file, dtype=np.int, mode='r', shape=(sizes[idx], 1)) for idx, val_file in
                        enumerate(val_files)]
            concat_size = np.sum(sizes)

            keys = np.memmap(args.concat_file_path + '/keys.npy', dtype=np.float16, mode='w+',
                            shape=(concat_size, args.dimension))
            vals = np.memmap(args.concat_file_path + '/vals.npy', dtype=np.int, mode='w+', shape=(concat_size, 1))

            cur_size = 0
            for idx, size in enumerate(sizes):
                logger.info('write {} to {}'.format(cur_size, cur_size + size))
                keys[cur_size: cur_size + size, :] = key_list[idx][:, :]
                vals[cur_size: cur_size + size, :] = val_list[idx][:, :]
                cur_size += size

            exit()

    if args.dstore_fp16:
        logger.info('load dstore fp16' + " " + str(args.dstore_size)+ " "+ str(args.dimension))
        keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r',
                            shape=(args.dstore_size, args.dimension))
        # vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
    else:
        keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r',
                        shape=(args.dstore_size, args.dimension))
        # vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

    logger.info('done.')
    if not os.path.exists(args.faiss_index+".trained"):
        # Initialize faiss index
        quantizer = faiss.IndexFlatIP(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
            args.ncentroids, args.code_size, 8)
        index.nprobe = args.probe

        logger.info('Training Index')
        np.random.seed(args.seed)
        num_training_point = 1000000
        random_sample = np.random.choice(np.arange(keys.shape[0]), size=[min(num_training_point, keys.shape[0])], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(keys[random_sample].astype(np.float32))
        logger.info('Training took {} s'.format(time.time() - start))

        logger.info('Writing index after training')
        start = time.time()
        faiss.write_index(index, args.faiss_index+".trained")
        logger.info('Writing index took {} s'.format(time.time()-start))

    logger.info('Adding Keys')
    index = faiss.read_index(args.faiss_index+".trained")
    start = args.starting_point
    start_time = time.time()
    while start < args.dstore_size:
        end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            logger.info('Added {} tokens so far'.format(start))
            logger.info('Writing Index {}'.format(start))
            faiss.write_index(index, args.faiss_index)

    logger.info("Adding total {} keys".format(start))
    logger.info('Adding took {} s'.format(time.time() - start_time))
    logger.info('Writing Index')
    start = time.time()
    faiss.write_index(index, args.faiss_index)
    logger.info('Writing index took {} s'.format(time.time()-start))

if __name__ == "__main__":
    train()
