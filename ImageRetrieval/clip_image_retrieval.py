import sys
import torch
import clip
import json
import os
import tarfile
import numpy as np
import logging
import train_datastore_gpu
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image, TarIO
from line_profiler import LineProfiler
from fairseq.modules.knn_datastore import KNN_Dstore
from fairseq.modules.visualization_html import VisualizationHtml

import time

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("clip.image.retrieval")

parser = ArgumentParser("ImageRetrieval Args")
parser.add_argument("--clip", type=str, default="RN50x16")
parser.add_argument("--mount", type=str, default="/path/to/blob")
parser.add_argument("--ifp", type=str, default="/valm/image_features")
parser.add_argument("--image_data_path", type=str, default="/valm/image/laion_all")
parser.add_argument("--tar_id_start", type=int, default=0)
parser.add_argument("--tar_id_end", type=int, default=19999) # 0-19999, 20000 in total
parser.add_argument("--n_gpus", type=int, default=63)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--portion", type=int, default=62) # portion: [0 : n_gpus-1]
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--verify_retriever", action="store_true")
parser.add_argument("--save_image_datastore", action="store_true")
parser.add_argument("--save_image_normalized", action="store_true")
parser.add_argument("--dstore_mmap", type=str, default=None)
parser.add_argument("--use_gpu_to_search", action="store_true")
parser.add_argument("--decoder_embed_dim", type=int, default=768)
parser.add_argument("--dstore_size", type=int, default=20000000)
parser.add_argument("--dstore_fp16", action="store_true")
parser.add_argument("--dstore_filename", type=str, default="")
parser.add_argument("--probe", type=int, default=32)
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--move_dstore_to_mem", action="store_true")

args = parser.parse_args()

class LaionDatasetReader(object):
    def __init__(self, id_start=0, id_end=0, n_gpus=24, image_data_path=None, missing_id=None):
        self.id_start = id_start
        self.id_end = id_end
        self.image_data_path = image_data_path
        self.n_gpus = n_gpus
        self.missing_id = missing_id
        self.partition_size = self.get_partition_size()

    def get_partition_size(self):
        # the number of tar for each gpu to process
        if self.missing_id is None:
            i = (self.id_end - self.id_start) // self.n_gpus
        else:
            i = len(self.missing_id) // self.n_gpus
        print(f"Partition size is {i}")
        return i

    def get_tar_portion(self, portion):
        if self.missing_id is None:
            portion_start_id = self.id_start + portion * self.partition_size
            portion_end_id = min(self.id_start + (portion + 1) * self.partition_size, self.id_end)
            if portion == self.n_gpus -1:
                portion_end_id = self.id_end
            tar_name_lst = []
            for i in range(portion_start_id,portion_end_id):
                tar_name_lst.append(f"{str(i).zfill(5)}.tar")
        else:
            portion_start_id = portion * self.partition_size
            portion_end_id = min((portion + 1) * self.partition_size, self.id_end)
            tar_name_lst = []
            for i in range(portion_start_id, portion_end_id):
                tar_name_lst.append(f"{str(self.missing_id[i]).zfill(5)}.tar")
        return tar_name_lst

    def get_image_batches(self, portion):
        tar_name_lst = self.get_tar_portion(portion)
        print(f"processing {len(tar_name_lst)} tar files in portion {portion} with tar list {tar_name_lst}")
        image_name_batches, tar_name_batches, global_ids = [], [], []
        for i in tqdm(range(len(tar_name_lst))):
            tar_file_name = tar_name_lst[i]
            tar_name_batch = self.get_image_batch(tar_file_name)
            tar_name_batches.append(tar_name_batch)
            global_ids.append(tar_file_name.replace(".tar",""))
        assert len(tar_name_batches) == len(global_ids)
        return tar_name_batches, global_ids

    def get_image_batch(self,tar_file_name):
        tar_file_path = f"{self.image_data_path}/{tar_file_name}"
        return tar_file_path




class ImageRetriever(object):
    def __init__(self, args=args, ifp=None, pth=None, Laion=True, loadLaionFeature=False, t_clip="RN50x16",reverse=False, batch_size=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(t_clip, self.device)
        self.image_path = pth
        self.image_feature_path = ifp
        self.batch_size = batch_size
        self.reverse = reverse
        self.args = args
        self.knn_datastore = None
        if ifp is None:
            print('Image feature path must be provided')
            raise ValueError
        if Laion:
            print('Build and save save image features in shards.')
            if loadLaionFeature:
                self.image_features, self.image_set = self.load_sharded_image_features(ifp)
        else:
            self.image_set = self.load_images()
            if ifp is not None and os.path.isfile(ifp):
                self.image_features, self.image_set = self.load_image_features(ifp)
                print(f"{len(self.image_set)} images loaded from {ifp}")
            else:
                ifp = "./data/image_features.pt" if ifp is None else ifp
                self.image_features = self.build_image_features()
                self.save_image_features(self.image_features, self.image_set, ifp)
        if args.dstore_mmap is not None:
            # init the mmap file for storing keys and values
            if not os.path.exists(args.dstore_mmap):
                os.makedirs(args.dstore_mmap)
            self.dstore_size = args.dstore_size
            self.decoder_embed_dim = args.decoder_embed_dim
            self.dstore_idx = 0
            self.dstore_fp16 = args.dstore_fp16
            self.dstore_mmap = args.dstore_mmap
            if args.save_image_datastore:
                if args.dstore_fp16:
                    logger.info('Saving fp16')
                    self.dstore_keys = np.memmap(os.path.join(self.dstore_mmap, 'keys.npy'), dtype=np.float16, mode='w+', shape=(self.dstore_size, self.decoder_embed_dim))
                    self.dstore_vals = np.memmap(os.path.join(self.dstore_mmap, 'vals.npy'), dtype=int, mode='w+', shape=(self.dstore_size, 1))
                else:
                    logger.info('Saving fp32')
                    self.dstore_keys = np.memmap(os.path.join(self.dstore_mmap, 'keys.npy'), dtype=np.float32, mode='w+', shape=(self.dstore_size, self.decoder_embed_dim))
                    self.dstore_vals = np.memmap(os.path.join(self.dstore_mmap, 'vals.npy'), dtype=np.int, mode='w+', shape=(self.dstore_size, 1))

    def load_images(self):
        i_set = []
        i_names = os.listdir(self.image_path)
        for i_n in i_names:
            i_set.append(i_n)
        print(f"Loading {len(i_set)} images in total.")
        return i_set

    def build_image_features(self):
        print(f"Build features for {len(self.image_set)} images...")
        batch_size, counter = self.batch_size, 0
        batch_image = []
        all_i_features = []
        # Prepare the inputs
        for i_n in tqdm(self.image_set):
            counter += 1
            i_input = self.preprocess(Image.open(f"{self.image_path}{i_n}")).unsqueeze(0).to(self.device)
            batch_image.append(i_input)
            if counter % batch_size == 0 or counter >= len(self.image_set):
                batch_image = torch.cat(batch_image,dim=0)
                with torch.no_grad():
                    i_feature = self.model.encode_image(batch_image)
                i_feature /= i_feature.norm(dim=-1, keepdim=True)
                all_i_features.append(i_feature.squeeze().to('cpu'))
                batch_image = []
        returned_image_features = torch.cat(all_i_features)
        return returned_image_features

    def save_image_features(self, image_feats, image_names, pth_to_save):
        assert len(image_feats) == len(image_names)
        print(f"Save {len(image_names)} image features at {pth_to_save}...")
        torch.save({'image_feats':image_feats, 'image_names':image_names}, pth_to_save)
        print(f"Done.")

    def load_image_features(self, pth_to_save):
        print(f"Load image features from {pth_to_save}...")
        checkpoint = torch.load(pth_to_save)
        return checkpoint['image_feats'], checkpoint['image_names']

    def load_sharded_image_features(self, pth_to_save, shards=5):
        n_shards = shards
        if os.path.exists(pth_to_save):
            print(f"Load {n_shards} sharded image features from {pth_to_save}...")
            pt_files = os.listdir(pth_to_save)
            image_features, image_names = [], []
            for pt_names in pt_files[:n_shards]:
                print(f"Loading from {pt_names}")
                checkpoint = torch.load(f"{pth_to_save}/{pt_names}")
                image_features.append(checkpoint['image_feats'])
                image_names += checkpoint['image_names']
            image_features = torch.cat(image_features)
            return image_features, image_names
        else:
            print(f"{pth_to_save} is not a valid path")
            raise FileNotFoundError

    def load_portion_image_features(self, pth_to_save, portion):
        if os.path.exists(pth_to_save):
            print(f"Load portion {portion} sharded image features from {pth_to_save}...")
            pt_files = os.listdir(pth_to_save)
            image_features, image_names = [], []
            pt_names = f"{pth_to_save}/img_features_{str(portion).zfill(5)}.pt"
            print(f"Loading from {pt_names}")
            checkpoint = torch.load(pt_names)
            image_features.append(checkpoint['image_feats'])
            image_names += checkpoint['image_names']
            image_features = torch.cat(image_features)
            return image_features, image_names
        else:
            print(f"{pth_to_save} is not a valid path")
            raise FileNotFoundError

    def get_text_features(self, text):
        text_inputs = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_laion_image_features(self, tar_batch, global_id_batch):
        def save_fun(i):
            tar_path, global_id = tar_batch[i], global_id_batch[i]

            file_path = f"{self.image_feature_path}/img_features_{global_id}.pt"
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 13000000:
                print(f"feature img_features_{global_id}.pt exists, continue to next image")
                return

            batch_size, counter = self.batch_size, 0
            batch_image = []
            all_i_features, returned_image_features, returned_image_names = [], [], []
            # Prepare the inputs
            if os.path.exists(tar_path):
                print(f"reading images in {tar_path}")
                with tarfile.open(tar_path, "r") as file:
                    image_files = {}
                    for i_n in file.getmembers():
                        if '.jpg' in i_n.name:
                            image_files[i_n.name] = Image.open(file.extractfile(i_n))
                    for i_n, i_f in tqdm(image_files.items()):
                        counter += 1
                        i_input = self.preprocess(i_f).unsqueeze(0).to(self.device)
                        batch_image.append(i_input)
                        returned_image_names.append(i_n)
                        if counter % batch_size == 0 or counter >= len(image_files):
                            batch_image = torch.cat(batch_image, dim=0)
                            with torch.no_grad():
                                i_feature = self.model.encode_image(batch_image)
                            # i_feature /= i_feature.norm(dim=-1, keepdim=True)
                            all_i_features.append(i_feature.to('cpu'))
                            batch_image = []

                    returned_image_features = torch.cat(all_i_features)
                    print(len(returned_image_features),len(returned_image_names))
                    self.save_image_features(returned_image_features, returned_image_names, f"{self.image_feature_path}/img_features_{global_id}.pt")
            else:
                print(f"tar path {tar_path} not exits.")
        if self.reverse:
            for tar_i in range(len(tar_batch)-1,0,-1):
                save_fun(tar_i)
        else:
            for tar_i in range(len(tar_batch)):
                save_fun(tar_i)

    def single_dot_retrieval(self, text, text_f, image_f):
        # Pick the top 5 most similar labels for the image
        N = 50000
        n_split = (len(image_f) // N) + 1
        similarity = []
        for i in range(n_split):
            i_f = image_f[i * N: (i + 1) * N].to(self.device) if (i + 1) * N < len(
                image_f) else image_f[i * N:].to(self.device)
            similarity.append((100 * i_f @ text_f.T).T.softmax(dim=-1).squeeze())

        similarity = torch.cat(similarity)
        values, indices = similarity.topk(5) if len(self.image_set) >= 5 else similarity.topk(len(self.image_set))
        # Print the result
        print(f"\nTop predictions of '{text}':\n")
        for value, index in zip(values, indices):
            print(f"{index} {self.image_set[index]}:  {100 * value.item():.2f}%")
        return similarity

    def retrieve(self, text):
        text_f = self.get_text_features(text)
        similarity = self.single_dot_retrieval(text, text_f, self.image_features)
        return similarity

    def prefix_retrieve(self, text):
        tokens = text.strip().split(' ')
        similarities = []
        for i in range(len(tokens)):
            prefix = ' '.join(tokens[:i+1])
            text_f = self.get_text_features(prefix)
            similarity = self.single_dot_retrieval(prefix, text_f, self.image_features)
            similarities.append(similarity)
        return similarities

    def get_images_and_features(self):
        return self.image_set, self.image_features

    def bulid_faiss_index_from_features(self, image_features, image_names):
        # image_features.shape = N x D, i.e. number of image x feature dimension, cpu pytorch tensor
        current_batch_count = image_features.shape[0]
        if self.dstore_idx + current_batch_count > self.dstore_size:
            reduce_size = self.dstore_size - self.dstore_idx
            image_features = image_features[:reduce_size]
            image_names = image_names[:reduce_size]
        else:
            reduce_size = current_batch_count
        
        if self.dstore_fp16:
            self.dstore_keys[self.dstore_idx:reduce_size + self.dstore_idx] = image_features.detach().cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:reduce_size + self.dstore_idx] = image_names.unsqueeze(-1).cpu().numpy().astype(np.int)
        else:
            self.dstore_keys[self.dstore_idx:reduce_size + self.dstore_idx] = image_features.detach().cpu().numpy().astype(np.float32)
            self.dstore_vals[self.dstore_idx:reduce_size + self.dstore_idx] = image_names.unsqueeze(-1).cpu().numpy().astype(np.int)
        self.dstore_idx += reduce_size
        logger.info("dstore_idx: " + str(self.dstore_idx))
        if self.dstore_idx >= self.dstore_size:
            logger.info('much more than dstore size break')
            exit(0)
        
    def train_faiss_index(self):
        train_datastore_gpu.train(dstore_mmap=self.dstore_mmap, dstore_size=self.dstore_idx, \
                                dstore_fp16=self.dstore_fp16)

    def setup_faiss(self):
        self.knn_datastore = KNN_Dstore(self.args)

    def faiss_retrieve(self, text):
        text_feature = self.get_text_features(text).view(1, 1, -1).type(torch.float32)
        search_start = time.time()
        knn_search_result = self.knn_datastore.retrieve(text_feature)
        print("retrieve tasks {:2f}s".format(time.time() - search_start))
        knn_dists = knn_search_result['distance'].cpu().numpy().tolist()[0][0]  # [batch, seq len, k]  # we need do sort
        knn_index = knn_search_result['knn_index']
        image_name = self.knn_datastore.vals[knn_index.detach().cpu()].tolist()[0][0]
        name = [i[0] for i in image_name]
        return (name, knn_dists)

if __name__ == '__main__':

    miss_id = None

    LaionData = LaionDatasetReader(id_start=args.tar_id_start, id_end=args.tar_id_end,
                                   n_gpus=args.n_gpus, image_data_path=f"{args.mount}{args.image_data_path}",
                                   missing_id=miss_id)
    imageRetriever = ImageRetriever(ifp=f"{args.mount}{args.ifp}", t_clip=args.clip, loadLaionFeature=args.verify,
                                    pth=f"{args.mount}{args.image_data_path}", Laion=True, reverse=args.reverse,
                                    batch_size=args.batch_size, args=args)

    if args.verify:
        # Retrieve from the first 5 shards to verify <feature-name> mappings
        imageRetriever.retrieve("A cute cat")  # 000048808.jpg √
        imageRetriever.retrieve("A cute dog")  # 000032573.jpg √
    elif args.save_image_datastore:
        for idx in range(1, 20000):
            file_name = 'img_features_{:05d}.pt'.format(idx)
            file_path = os.path.join(f"{args.mount}{args.ifp}", file_name)
            if not os.path.exists(file_path):
                continue
            logger.info(file_name)
            saved_image = torch.load(file_path)
            saved_image_features = saved_image['image_feats']
            if args.save_image_normalized:
                saved_image_features /= saved_image_features.norm(dim=-1, keepdim=True)
            saved_image_names = saved_image['image_names']
            saved_image_names = torch.tensor([int(name.split('.')[0]) for name in saved_image_names])
            imageRetriever.bulid_faiss_index_from_features(saved_image_features, saved_image_names)
        imageRetriever.dstore_size = imageRetriever.dstore_idx     
    elif args.verify_retriever:
        imageRetriever.setup_faiss()
        text_list = ["The color of parsley is"]
        for text in text_list:
            image_name, dists = imageRetriever.faiss_retrieve(text)
            print(image_name)
            print(dists)

            html = VisualizationHtml(text,image_name,dists,image_data_path=f"{args.mount}{args.image_data_path}")
            html.save_report("./html/reports.html")

    else:
        LainTarNameBatch, LainImageID = LaionData.get_image_batches(args.portion)
        imageRetriever.get_laion_image_features(LainTarNameBatch, LainImageID)
