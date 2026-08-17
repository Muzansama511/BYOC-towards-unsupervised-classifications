from collections import deque
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
import numpy as np
import os
from tqdm import tqdm
from functools import partial
from time import time
import pyarrow.parquet as pq

# test for tiktoken local cache (see https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer if want to run offline)
# tk_cache_dir = os.environ.get('TIKTOKEN_CACHE_DIR')

# assert os.path.exists(os.path.join(tk_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# NOTE: OPTIMS:
# Read from Numpy

# from time import time
## Example usage:
# s = time()
# df = pd.read_parquet('/mnt/d/OpenWebText/plain_text/train-00000-of-00080.parquet')
# print(df)
# print(f"Time taken to read parquet file: {time() - s:.4f} seconds")
# exit()
'''
MASK is '♥' <Alt + 147896323>
'''

def get_tiktokenizer():
    base = tiktoken.get_encoding("cl100k_base")
    # print(base._special_tokens)
    custom_special_tokens = {
        **base._special_tokens,
        "♣": 100277,   # CLS
        "♦": 100278,   # SEP
        "♥": 100279,   # MASK
        "♠": 100280,   # PAD
    }

    enc = tiktoken.Encoding(
        name="cl100k_worldmodel",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=custom_special_tokens,
    )

    CLS_ID = enc._special_tokens["♣"]
    SEP_ID = enc._special_tokens["♦"]
    TOTAL_VOCAB_SIZE = enc.n_vocab
    SPECIAL_VOCAB_SIZE = len(enc._special_tokens)
    return enc, {"CLS_ID": CLS_ID, "SEP_ID": SEP_ID, "MASK_ID": enc._special_tokens["♥"], "PAD_ID": enc._special_tokens["♠"], "VOCAB_SIZE": TOTAL_VOCAB_SIZE - SPECIAL_VOCAB_SIZE, "SPECIAL_VOCAB_SIZE" : SPECIAL_VOCAB_SIZE}

####### Test for the tokenizer
# get_tiktokenizer()

def make_dataset_to_bin(dataset_dir="/mnt/d/OpenWebText/plain_text"):

    parquet_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.parquet')]
    out_parent = os.path.dirname(dataset_dir) if dataset_dir[-1] != '/' else os.path.dirname(os.path.dirname(dataset_dir))

    block_size = 512
    content_size = block_size - 2   # leaves room for CLS + SEP per chunk
    val_fraction = 0.05
    chunk_size = 2000                # texts processed per tokenize call
    num_threads = 4
    min_chunk_tokens = 32             # drop tiny leftover tail chunks (mostly padding, low signal)

    enc, meta = get_tiktokenizer()
    CLS_ID = meta["CLS_ID"]
    SEP_ID = meta["SEP_ID"]

    train_bin_path = os.path.join(out_parent, "train.bin")
    val_bin_path = os.path.join(out_parent, "val.bin")

    train_offsets = [0]
    val_offsets = [0]
    train_total = 0
    val_total = 0
    doc_counter = 0

    with open(train_bin_path, "wb") as f_train, open(val_bin_path, "wb") as f_val:
        for pf in tqdm(parquet_files, desc="parquet files"):
            table = pq.read_table(pf, columns=["text"])
            texts = table.column("text").to_pylist()
            del table

            for start in tqdm(range(0, len(texts), chunk_size), desc=f"chunks in {os.path.basename(pf)}", leave=False):
                text_chunk = texts[start:start + chunk_size]
                body_ids_batch = enc.encode_ordinary_batch(text_chunk)

                for body_ids in body_ids_batch:
                    if not body_ids:
                        continue

                    # split this single document into multiple non-overlapping content_size pieces
                    for i in range(0, len(body_ids), content_size):
                        piece = body_ids[i:i + content_size]
                        if len(piece) < min_chunk_tokens:
                            continue  # skip negligible tail fragments

                        seq = np.array([CLS_ID] + piece + [SEP_ID], dtype=np.uint32)

                        if doc_counter % int(1 / val_fraction) == 0:
                            f_val.write(seq.tobytes())
                            val_total += len(seq)
                            val_offsets.append(val_total)
                        else:
                            f_train.write(seq.tobytes())
                            train_total += len(seq)
                            train_offsets.append(train_total)

                        doc_counter += 1
                print(f"train: {train_total} tokens, {len(train_offsets)-1} chunks -> {train_total*4/1e6:.1f} MB")
                print(f"val:   {val_total} tokens, {len(val_offsets)-1} chunks -> {val_total*4/1e6:.1f} MB")

                del text_chunk, body_ids_batch

            del texts

    np.save(os.path.join(out_parent, "train_offsets.npy"), np.array(train_offsets, dtype=np.int64))
    np.save(os.path.join(out_parent, "val_offsets.npy"), np.array(val_offsets, dtype=np.int64))

########### To make the bin file from dataset, done only once.
# make_dataset_to_bin()
# exit()
###########

# class OpenWebTextDataset(Dataset):
#     def __init__(self, verbose = True, return_tokens = False):
#         '''
#         cutoff is the size of the text sample
#         '''
#         self.parent_dir = '/mnt/d/OpenWebText/plain_text/'
#         self.return_tokens = return_tokens
#         # metadata = {} # stores how many rows each parquet file has
#         if ('metadata.csv' not in os.listdir(self.parent_dir)):
#             metadata_num_rows = []
#             if verbose:
#                 print('Making metadata.csv...')
#             for file in tqdm(os.listdir(self.parent_dir), desc='Processing files', disable = not verbose):
#                 if file.endswith('.parquet'):
#                     file_path = os.path.join(self.parent_dir, file)
#                     df = pd.read_parquet(file_path)
#                     metadata_num_rows.append(len(df))
#             metadata = {'file': [file for file in os.listdir(self.parent_dir) if file.endswith('.parquet')], 'num_rows': metadata_num_rows}
#             metadata_df = pd.DataFrame(metadata)
#             metadata_df.to_csv(os.path.join(self.parent_dir, 'metadata.csv'), index=False)
#         else:
#             self.df = pd.read_csv(os.path.join(self.parent_dir, 'metadata.csv'))
#             metadata = {'file': self.df['file'].tolist(), 'num_rows': self.df['num_rows'].tolist()}
#         self.master = metadata
#         self.master_cumsum = np.cumsum(self.master['num_rows'])
#         self.enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, GPT-3.5-turbo
#     def __len__(self):
#         return sum(self.master['num_rows'])

#     def __getitem__(self, idx):
#         # Find which file the index belongs to
#         s_loc = time()
#         file_idx = np.searchsorted(self.master_cumsum, idx, side='right')
#         cumulative_rows = self.master_cumsum[file_idx - 1] if file_idx > 0 else 0
#         # for i, num_rows in enumerate(self.master['num_rows']):
#         #     if idx < cumulative_rows + num_rows:
#         #         file_idx = i
#         #         break
#         #     cumulative_rows += num_rows

#         # Read the specific row from the corresponding file
#         file_path = os.path.join(self.parent_dir, self.master['file'][file_idx])
#         df = pd.read_parquet(file_path)
        
#         if self.return_tokens:
#             output = self.enc.encode(df.iloc[idx - cumulative_rows].values[0])
#         else:
#             output = df.iloc[idx - cumulative_rows].values[0]
#         print(f"Time taken to read parquet file: {time() - s_loc:.4f} seconds")
#         return output


class OpenWebTextDatasetFromBin(Dataset):
    def __init__(self, train_or_val='train', perc_to_use = 0.9):
        '''
        cutoff is the size of the text sample
        '''
        self.file_cache = deque(maxlen=10)  # Cache for 10 files
        self.parent_dir = '/mnt/d/OpenWebText/'
        self.perc_to_use = perc_to_use
        assert 0<perc_to_use<=1, 'perc_to_use Should be a valid percentage'
        self.return_tokens = True # Can only return tokens
        if train_or_val == 'train':
            self.bin_path = os.path.join(self.parent_dir, 'train.bin')
            # print(self.bin_path)
            self.offsets = np.load(os.path.join(self.parent_dir, 'train_offsets.npy'))
    
        else:    
            self.bin_path = os.path.join(self.parent_dir, 'val.bin')
            self.offsets = np.load(os.path.join(self.parent_dir, 'val_offsets.npy'))
        
        self.data = np.memmap(self.bin_path, dtype=np.uint32, mode="r")
        
        # self.enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, GPT-3.5-turbo
    def __len__(self):
        return int((len(self.offsets)-1)*self.perc_to_use)

    def __getitem__(self, idx):
        # Find which file the index belongs to
        # s_loc = time()
        start = self.offsets[idx]
        end = self.offsets[idx+1]
        
        # print(start, end)
        # print(f"Time taken to read file: {time() - s_loc:.4f} seconds")        
        return self.data[start:end].copy()
        # data_bits = 32 # uint32
        
        
        # with open(self.bin_path, "rb") as f:
        #     f.seek(start * (data_bits // 8))  # Move to the start position in bytes
        #     output = np.fromfile(f, dtype=np.uint32, count=chunk)
        # # print(f"Time taken to read file: {time() - s_loc:.4f} seconds")
        # return output

######### Test for speed improvement 
# # d = OpenWebTextDataset(verbose = False, return_tokens = True) # 1.70s/it
# d = OpenWebTextDatasetFromBin() # 300+it/s, after memmap: ~80000it/s

# s = time()
# for i in tqdm(range(len(d))):
#     output = (d[i])
#     # print(output)
# print('Total time for reading all the data', time()-s, " seconds")
# # exit()
#########

# def custom_collate(batch, max_length =  128):
#     # Add masking with 80 10 10 (80% of eddited masked, 10% changed and 10% unchanged)
#     # Find the maximum length of the sequences in the batch
#     MASKING_PROB = 0.2
#     s_loc = time()
#     _, meta = get_tiktokenizer()
#     vocab_size = meta['VOCAB_SIZE']
#     mask_id = meta['MASK_ID']
#     pad_id = meta['PAD_ID']
#     # print(meta)
#     reformed_batch = []
#     for b in batch:
#         if len(b)>(max_length):
#             x = random.randint(0,(len(b)-1-max_length))+1
#             reformed_b = np.concat([b[0:1], b[x:x+(max_length-1)]])
#         elif len(b)<(max_length):
#             reformed_b = np.concat([b[0:-1], np.array([pad_id]*((max_length)-len(b)), dtype=np.uint32), b[-1:]])
#         else:
#             reformed_b = b.copy()
#         reformed_batch.append(reformed_b)
#     reformed_batch = np.array(reformed_batch)
#     # print('reformed_batch shape: ', reformed_batch.shape)
#     # Make masking
#     non_pad_tokens = (reformed_batch!=pad_id)
#     mask = np.clip((np.random.rand(*reformed_batch.shape)<MASKING_PROB) + (~non_pad_tokens),0,1)
#     edit = (np.random.rand(*mask.shape) * (mask) * (non_pad_tokens))
#     mask_edit = (edit > 0.2)
#     change_edit = (edit > 0.1) & (edit <= 0.2)
    
#     output = torch.from_numpy(reformed_batch.copy()).to(torch.int64)
    
#     mask = torch.from_numpy(mask).to(torch.bool)
#     mask_edit = torch.from_numpy(mask_edit.astype(np.bool_)).to(torch.bool)
#     output.masked_fill_(mask_edit, mask_id)
    
#     change_edit = torch.from_numpy(change_edit.astype(np.bool_)).to(torch.bool)
#     random_tokens = torch.randint(0, vocab_size, output.shape, dtype=torch.int64)
#     output = torch.where(change_edit, random_tokens, output.to(torch.int64)).to(output.dtype)
    
#     reformed_batch = torch.from_numpy(reformed_batch).to(torch.int64)
#     print(f"Time taken for collate: {time() - s_loc:.4f} seconds")
#     return {"input":reformed_batch, "output":output, "mask":mask, "consider_for_loss":((mask) & (non_pad_tokens))}

# # Get these ONCE, not every collate call
_, meta = get_tiktokenizer()

VOCAB_SIZE = meta["VOCAB_SIZE"]
MASK_ID = meta["MASK_ID"]
PAD_ID = meta["PAD_ID"]


def custom_collate(batch, max_length=128, shuffle=False):

    s = time()
    batch_size = len(batch)

    # ---------------------------------------------------------
    # 1. Allocate output directly
    # ---------------------------------------------------------

    # uint32 is enough for token IDs and saves memory
    tokens = np.full(
        (batch_size, max_length),
        PAD_ID,
        dtype=np.uint32
    )

    # ---------------------------------------------------------
    # 2. Crop / pad sequences
    # ---------------------------------------------------------

    for i, seq in enumerate(batch):

        seq_len = len(seq)
        if seq_len > max_length:
            # Keep first token (e.g. BOS)
            # Randomly sample max_length - 1 tokens after it
            if shuffle==False:
                start = 1
            else:
                start = np.random.randint(
                    1,
                    seq_len - max_length + 1
                )

            tokens[i, 0] = seq[0]
            tokens[i, 1:] = seq[start:start + max_length - 1]

        else:

            # Keep original sequence
            tokens[i, :seq_len] = seq

    # ---------------------------------------------------------
    # 3. Convert to torch ONCE
    # ---------------------------------------------------------

    input_ids = torch.from_numpy(tokens).long()

    # Valid tokens
    non_pad = input_ids != PAD_ID

    # ---------------------------------------------------------
    # 4. Select tokens to corrupt
    # ---------------------------------------------------------

    # 10% of non-padding tokens
    r = torch.rand(
                (batch_size, max_length),
                device=input_ids.device
            )
    
    mask = (
        r < 0.25 # should mask 10% values
    ) & non_pad
    
    if mask.sum() == 0:
        
        mask[:, 1] = True
    # print(mask)
    # ---------------------------------------------------------
    # 5. 80 / 10 / 10 masking
    # ---------------------------------------------------------

    rand = torch.rand(
        (batch_size, max_length),
        device=input_ids.device
    )

    # 80% -> MASK
    mask_tokens = mask & (rand < 0.8) # 80% rn

    # 10% -> random token
    random_tokens = mask & (rand >= 0.8) & (rand < 0.9)

    # 10% -> unchanged
    # mask & (rand >= 0.9)

    output = input_ids.clone()

    # 80% MASK
    output.masked_fill_(
        mask_tokens,
        MASK_ID
    )

    # 10% random token
    random_ids = torch.randint(
        0,
        VOCAB_SIZE,
        output.shape,
        dtype=torch.long
    )

    output[random_tokens] = random_ids[random_tokens]

    # ---------------------------------------------------------
    # 6. Return
    # ---------------------------------------------------------
    # print(f'Time taken for collate: {time() - s:.4f} seconds')
    return {
        "input": input_ids,
        "output": output,
        "mask": mask,
        "consider_for_loss": mask,
    }
########## Test for collate
# out = custom_collate([list(np.arange(20))], max_length = 10)
# print(out)
# out = custom_collate([list(np.arange(20))], max_length = 19)
# print(out)
# out = custom_collate([list(np.arange(20))], max_length = 20)
# print(out)
# out = custom_collate([list(np.arange(20))], max_length = 30)
# print(out)
# exit()

def get_dataloader(train_or_val = "train", batch_size = 16, num_workers = 4, max_length = 64, perc_to_use = 1.0, shuffle = False):
    dataset = OpenWebTextDatasetFromBin(train_or_val=train_or_val, perc_to_use = perc_to_use)
    partial_custom_collate = partial(custom_collate, max_length=max_length, shuffle = shuffle)  # Set your desired max_length here
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=partial_custom_collate, num_workers=num_workers)
    return dataloader

###### Test for dataloader
# d = get_dataloader(batch_size=4, num_workers=4, max_length=128)
# s = time()
# idx = 0
# for batch in tqdm(d):
#     mask = batch['mask']
#     print('sum', torch.sum(mask))
#     # print(batch)
#     idx+=1
#     if idx==100:
#         break
    # print(f"Time taken for batch: {time() - s:.4f} seconds")
    # s = time()
    # break
# d = get_dataloader(train_or_val = 'val', batch_size=16, num_workers=4, max_length=128)
# s = time()
# idx = 0
# for batch in tqdm(d):
#     idx+=1
#     if idx==100:
#         break
#     # print(batch)
#     # print(f"Time taken for batch: {time() - s:.4f} seconds")
#     # s = time()
#     # break
######

# def custom_collate(batch, return_tokens = False,max_length =  64):
#     # Find the maximum length of the sequences in the batch
#     s_loc = time()
#     max_length = max(len(seq) for seq in batch)
#     max_length = min(max_length, max_length)  # Ensure we don't exceed the specified max_length

#     # Pad sequences to the maximum length
#     if return_tokens:
#         padded_batch = [seq + [77809] * (max_length - len(seq)) for seq in batch]
#     else:
#         padded_batch = [seq + '♥'* (max_length - len(seq)) for seq in batch]
#         return padded_batch
#     print(f"Time taken for collate: {time() - s_loc:.4f} seconds")
#     return torch.tensor(padded_batch, dtype=torch.long)

# def get_dataloader(batch_size = 16, num_workers = 4, return_tokens = False, max_length = 64):
#     dataset = OpenWebTextDataset(verbose = True, return_tokens = return_tokens)
#     partial_custom_collate = partial(custom_collate, return_tokens=return_tokens, max_length=max_length)  # Set your desired max_length here
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=partial_custom_collate, num_workers=num_workers)
#     return dataloader

# dl = get_dataloader(batch_size=16, num_workers=0, return_tokens=True, max_length=64)

# count = 0
# s = time()
# for batch in dl:
#     print(batch)
#     count += 1
#     e = time()
#     print(f"Time taken for batch {count}: {e - s:.4f} seconds")
#     s = time()
#     if count >= 2:
#         break  # Just to show the first batch


# enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, GPT-3.5-turbo

# # # print(enc.encode((b"\xff"))) 
# # # token_id = 99999
# # # token_bytes = enc.decode_single_token_bytes(token_id)
# # # print(token_bytes)
# # # print(enc._mergeable_ranks.get(b"\xff"))  # This will print the rank of the byte b"\xff" in the mergeable ranks
# # # print(enc._mergeable_ranks.get(b"\x00"))  # This will print the rank of the byte b"\x00" in the mergeable ranks
# # # print(enc.decode_single_token_bytes(enc._mergeable_ranks.get(b"\xff")))
# print(enc.encode('♥♥♥♥♥♥♥♥'))  # This will print the byte corresponding to the rank of b"\x00"
