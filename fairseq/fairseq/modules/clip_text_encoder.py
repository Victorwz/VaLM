import torch
import clip
import numpy as np
import os
from tqdm import tqdm

from clip.simple_tokenizer import SimpleTokenizer

class CLIP_Text_Encoder:
    def __init__(self, model_type="RN50x16"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_type, device=self.device)
        self.decoder = SimpleTokenizer()

    def encode_text(self, prev_output_tokens, context_length=77):
        batch_size, length = prev_output_tokens.shape
        input_tokens = prev_output_tokens.detach().cpu().numpy()
        input_tokens = input_tokens - 4
        input_tokens[np.where(input_tokens == -2)] = 49407 # deal with eos
        input_tokens[np.where(input_tokens == -3)] = 0 # deal with pad
        batch_features = []
        
        sot_token = self.decoder.encoder["<|startoftext|>"]
        eot_token = self.decoder.encoder["<|endoftext|>"]
        
        for sample in input_tokens:
            sub_sent_feature_list = []
            # sentence tokenization
            sentences = self.sentence_tokenizer(sample)

            for sent in sentences:
                sent_len = len(sent)
                sent_tokenize = [sot_token] + sent + [eot_token] + [0] * (context_length-sent_len-2)
                sent_tokenize = torch.tensor([sent_tokenize]).to(self.device)
                text_features = self.get_text_encoding(sent_tokenize)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                sub_sent_feature_list.append(text_features[0, 1:1+sent_len])
    
            whole_sent_feature = torch.cat(sub_sent_feature_list, dim=0)
            assert whole_sent_feature.shape[0] == length
            
            batch_features.append(whole_sent_feature)

        batch_features = torch.stack(batch_features).type(torch.float32)

        return batch_features

    def sentence_tokenizer(self, sample):
        sub_sentences = []
        line = sample.tolist()
        while len(line) > 75:
            try:
                eos_index = line.index(49407) # 49407 is eos
                if len(line[:(eos_index+1)]) < 76:
                    sub_sentences.append(line[:(eos_index+1)])
                    line = line[(eos_index+1):]
                    continue
            except ValueError:
                pass
            try:
                pad_index = line.index(0) # 0 is pad
                if len(line[:(pad_index+1)]) < 76:
                    sub_sentences.append(line[:(pad_index+1)])
                    line = line[(pad_index+1):]
                    continue
            except ValueError:
                pass
            try:
                stop_index = line.index(269) # 269 is .
                if len(line[:(stop_index+1)]) < 76:
                    sub_sentences.append(line[:(stop_index+1)])
                    line = line[(stop_index+1):]
                    continue
            except ValueError:
                pass     
            try:
                smc_index = line.index(282) # 282 is ;
                if len(line[:(smc_index+1)]) < 76:
                    sub_sentences.append(line[:(smc_index+1)])
                    line = line[(smc_index+1):]
                    continue
            except ValueError:
                pass
            try:
                smc_index = line.index(256) # 256 is !
                if len(line[:(smc_index+1)]) < 76:
                    sub_sentences.append(line[:(smc_index+1)])
                    line = line[(smc_index+1):]
                    continue
            except ValueError:
                pass
            try:
                colon_index = line.index(281) # 281 is :
                if len(line[:(colon_index+1)]) < 76:
                    sub_sentences.append(line[:(colon_index+1)])
                    line = line[(colon_index+1):]
                    continue
            except ValueError:
                pass
            try:
                comma_index = line.index(267) # 267 is ,
                if len(line[:(comma_index+1)]) < 76:
                    sub_sentences.append(line[:(comma_index+1)])
                    line = line[(comma_index+1):]
                    continue
            except ValueError:
                pass
            try:
                stop_quote_index = line.index(1081) # 1081 is ."
                if len(line[:(stop_quote_index+1)]) < 76:
                    sub_sentences.append(line[:(stop_quote_index+1)])
                    line = line[(stop_quote_index+1):]
                    continue
            except ValueError:
                pass
            try:
                stop_quote_index = line.index(3823) # 3823 is ,"
                if len(line[:(stop_quote_index+1)]) < 76:
                    sub_sentences.append(line[:(stop_quote_index+1)])
                    line = line[(stop_quote_index+1):]
                    continue
            except ValueError:
                pass
            try:
                stop_quote_index = line.index(2811) # 2811 is ".
                if len(line[:(stop_quote_index+1)]) < 76:
                    sub_sentences.append(line[:(stop_quote_index+1)])
                    line = line[(stop_quote_index+1):]
                    continue
            except ValueError:
                pass
            try:
                stop_quote_index = line.index(4332) # 4332 is ",
                if len(line[:(stop_quote_index+1)]) < 76:
                    sub_sentences.append(line[:(stop_quote_index+1)])
                    line = line[(stop_quote_index+1):]
                    continue
            except ValueError:
                pass
            try:
                bracket_index = line.index(1818) # 1818 is .)
                if len(line[:(bracket_index+1)]) < 76:
                    sub_sentences.append(line[:(bracket_index+1)])
                    line = line[(bracket_index+1):]
                    continue
            except ValueError:
                pass
            try:
                bracket_index = line.index(2361) # 2361 is ),
                if len(line[:(bracket_index+1)]) < 76:
                    sub_sentences.append(line[:(bracket_index+1)])
                    line = line[(bracket_index+1):]
                    continue
            except ValueError:
                pass
            try:
                and_index = line.index(270) # 270 is /
                if len(line[:(and_index+1)]) < 76:
                    sub_sentences.append(line[:(and_index+1)])
                    line = line[(and_index+1):]
                    continue
            except ValueError:
                pass
            try:
                dash_index = line.index(268) # 268 is -
                if len(line[:(dash_index+1)]) < 76:
                    sub_sentences.append(line[:(dash_index+1)])
                    line = line[(dash_index+1):]
                    continue
            except ValueError:
                pass
            # print("No seperations found in {}".format(line))
            sub_sentences.append(line[:75])
            line = line[75:]
            
        if line != []:
            sub_sentences.append(line)
        return sub_sentences

    def get_text_encoding(self, text):
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.model.text_projection

        return x      






