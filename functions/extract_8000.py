import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import pandas as pd

from Bio import SeqIO
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from keras.utils import pad_sequences


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


class ChunkDataset(Dataset):
    def __init__(self, text, labels, tokenizer, chunk_len=512, overlap_len=0):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.labels)

    def chunk_tokenizer(self, tokenized_data, targets):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = tokenized_data["input_ids"]
        previous_attention_mask = tokenized_data["attention_mask"]
        previous_token_type_ids = tokenized_data["token_type_ids"]
        remain = tokenized_data.get("overflowing_tokens")

        input_ids_list.append(
            torch.tensor(previous_input_ids, dtype=torch.long))
        attention_mask_list.append(
            torch.tensor(previous_attention_mask, dtype=torch.long))
        token_type_ids_list.append(
            torch.tensor(previous_token_type_ids, dtype=torch.long))
        targets_list.append(torch.tensor(targets, dtype=torch.long))

        if remain:  # if there is any overflowing tokens
            # remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain) + self.chunk_len)
            idxs = idxs[(self.chunk_len - self.overlap_len - 2)
                        ::(self.chunk_len - self.overlap_len - 2)]
            input_ids_first_overlap = previous_input_ids[
                                      -(self.overlap_len + 1):-1]

            start_token = [1]
            end_token = [2]

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = input_ids_first_overlap + remain[:idx]
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx - self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids) + 2
                attention_mask = np.ones(self.chunk_len)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = np.zeros(self.chunk_len)
                input_ids = start_token + input_ids + end_token
                if self.chunk_len - nb_token > 0:
                    padding = np.zeros(self.chunk_len - nb_token)
                    input_ids = np.concatenate([input_ids, padding])

                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                attention_mask_list.append(
                    torch.tensor(attention_mask, dtype=torch.long))
                token_type_ids_list.append(
                    torch.tensor(token_type_ids, dtype=torch.long))
                targets_list.append(torch.tensor(targets, dtype=torch.long))

        return ({
            'ids': input_ids_list,
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __getitem__(self, index):
        text = " ".join(str(self.text[index]).split())
        targets = self.labels[index]

        data = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.chunk_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        chunk_token = self.chunk_tokenizer(data, targets)
        return chunk_token


def chunk_collate_fn(batches):
    """
    Create batches for ChunkDataset
    """
    return [{key: torch.stack(value) for key, value in batch.items()} for batch
            in batches]


class MyDataset(Dataset):
    def __init__(self, df):
        self.X = df['SequenceID'].to_list()
        self.Y = df['Label']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y.iloc[index]


def load_seq(seq_path, kmer):
    sequence = []
    seq_label = []
    seq_content_list = []

    for seq_record in SeqIO.parse(seq_path, "fasta"):
        seq_label.append(str((seq_record.id).split(',')[0]))
        seq_origin = str(seq_record.seq.strip())
        seq_origin = seq_origin.upper().replace('T', 'U')
        sequence.append(seq2kmer(seq_origin.strip(), kmer))
    df = pd.DataFrame(data={'SequenceID': sequence, 'Label': seq_label})
    return df


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    # kmer = re.findall(r'\w{3}',seq)
    kmers = " ".join(kmer)
    return kmers


def kmer2seq(kmers):
    """
    Convert kmers to original sequence

    Arguments:
    kmers -- str, kmers separated by space.

    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


def vectorize_labels(all_labels):
    """
    :return: dict of vectorized labels per split and total number of labels
    """
    result = {}
    for split in all_labels:
        result[split] = np.array(all_labels[split])
    return result


def remove_special_token(embedding, attention_mask):
    transform = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len - 1]
        transform.append(seq_emd)
    transform_emb = np.vstack(transform)
    return transform_emb


def prepare_data(data_path, dataset_num, num_labels, kmer):
    """
    return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(data_path):
        raise Exception("Data path not found: {}".format(data_path))

    text_set = {'seq_to_extract': []}
    label_set = {'seq_to_extract': []}

    dataset_split = ['seq_to_extract']
    for each_item in dataset_split:
        wholepath = data_path + each_item + '.fasta'
        df_dataset = load_seq(wholepath, kmer)
        df_dataset["Label"] = df_dataset["Label"].apply(
            lambda x: list(map(int, x)))
        df_dataset = MyDataset(df_dataset)
        for item in df_dataset:
            # print(item[0])
            text_set[each_item].append(item[0])
            label_set[each_item].append(item[1])

    vectorized_labels = vectorize_labels(label_set)
    return text_set, vectorized_labels, num_labels


def create_dataloader(dataset_class, text_set, label_set, tokenizer, max_length,
                      batch_size, num_workers):
    """
    Create appropriate dataloaders for the given data
    """
    dataloaders = {}

    if 'seq_to_extract' in text_set.keys():
        split = 'seq_to_extract'
        dataset = dataset_class(text_set[split], label_set[split], tokenizer,
                                max_length)
        if isinstance(dataset, ChunkDataset):
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=chunk_collate_fn)
        else:
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)



    return dataloaders

def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores)+kmer-1])
    real_scores = np.zeros([len(attention_scores)+kmer-1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score

        real_scores = real_scores/counts
    else:
        pass

    return real_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="path to the data folder",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="path to the output folder",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="path to the model folder",
    )
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path

    classes = 7
    kmer = 3
    max_length = 512
    batch_size = 5
    num_workers = 0
    fixed_length = 8000

    mk_dir(output_path)

    tokenizer = BertTokenizer.from_pretrained(
        args.model_path, do_lower_case=False)
    text_set, label_set, num_labels = prepare_data(data_path, dataset_num,
                                                   classes, kmer)

    dataset_class = ChunkDataset
    dataloaders = create_dataloader(dataset_class, text_set, label_set,
                                    tokenizer, max_length, batch_size,
                                    num_workers)

    model = BertModel.from_pretrained(args.model_path,
                                      config=BertConfig.from_pretrained(
                                          args.model_path,
                                          output_attentions=True))

    
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()

    embedding = []
    print(dataloaders.keys())
    for each_id in ["seq_to_extract"]:
        if each_id == "seq_to_extract":
            split = "seq_to_extract"
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloaders[split], 0):
                    for each_item in data:
                        ids = each_item['ids'].to(device, dtype = torch.long)
                        mask = each_item['mask'].to(device, dtype = torch.long)
                        token_type_ids = each_item['token_type_ids'].to(device, dtype = torch.long) 
                        outputs = model(input_ids=ids,attention_mask=mask,output_hidden_states=True) 
                        hidden_states = outputs[2]  
                        hidden_states = (hidden_states[-1] + hidden_states[1]).cpu().numpy()
                        transform_emb = remove_special_token(hidden_states, mask)
                        embedding_pad = np.pad(transform_emb, ((0,fixed_length-transform_emb.shape[0]),(0,0)), 'mean')
                        embedding_pad = np.mean(embedding_pad, axis=1)
                        embedding.append(embedding_pad[:, np.newaxis])
    
    np.save(output_path + 'seq_to_extract' + '.npy',
            np.array(embedding))
    print(np.array(embedding).shape)

if __name__ == "__main__":
    main()




