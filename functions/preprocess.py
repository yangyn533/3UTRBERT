# extract 20% independent_test_set, do 5-fold split in remaining data
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
import argparse
import numpy as np
import csv

random.seed(42)

def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def extract_test_and_split_train_vali(input_path, save_path, kmer):
    input_file = input_path

    seq_list = []

    for seq_record in tqdm(SeqIO.parse(input_file, "fasta")):
      if len(str(seq_record.seq)) > 510:
        print(str(seq_record.seq))
        raise ValueError("The input sequence should not be longer than 510 nts.")
      seq_record.seq = seq_record.seq.upper()
      seq_list.append(seq_record)
    random.shuffle(seq_list)
    
    length = len(seq_list)
    test_len = length // 5
    independent_test_set = seq_list[:test_len]
    kmer_to_tsv_test = []
    for seq_record in independent_test_set:
        seq_record.seq = seq_record.seq.upper()
        final_kmer = seq2kmer(str(seq_record.seq), kmer)
        label = seq_record.id
        kmer_to_tsv_test.append([final_kmer, label])
    
    with open(save_path + "/test.tsv", 'w') as f1:
          tsv_w = csv.writer(f1, delimiter='\t')
          tsv_w.writerow(['sequence', 'label'])
          for row in kmer_to_tsv_test:
            tsv_w.writerow(row)


    remain_data = seq_list[test_len:]
    
    kf = KFold(n_splits=5,shuffle=True, random_state=42)  # initialize KFold
    i = 0
    for train_index, vali_index in kf.split(remain_data):  
        vali_set = []
        for index in vali_index:
          vali_set.append(remain_data[index])
        train_set = []
        for indexes in train_index:
          train_set.append(remain_data[indexes])
        
        kmer_to_tsv_train = []
        for seq_record in train_set:
          seq_record.seq = seq_record.seq.upper()
          final_kmer = seq2kmer(str(seq_record.seq), kmer)
          label = seq_record.id
          kmer_to_tsv_train.append([final_kmer, label])
        if not os.path.exists(save_path + "/fold{}".format(i)):
          os.mkdir(save_path + "/fold{}".format(i))
        with open(save_path + "/fold{}".format(i) + "/train.tsv", 'w') as f2:
          tsv_w = csv.writer(f2, delimiter='\t')
          tsv_w.writerow(['sequence', 'label'])
          for row in kmer_to_tsv_train:
            tsv_w.writerow(row)


        kmer_to_tsv_vali = []
        for seq_record in vali_set:
          seq_record.seq = seq_record.seq.upper()
          final_kmer = seq2kmer(str(seq_record.seq), kmer)
          label = seq_record.id
          kmer_to_tsv_vali.append([final_kmer, label])

        with open(save_path + "/fold{}".format(i) + "/dev.tsv", 'w') as f3:
          tsv_w = csv.writer(f3, delimiter='\t')
          tsv_w.writerow(['sequence', 'label'])
          for row in kmer_to_tsv_vali:
            tsv_w.writerow(row)
        i+=1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input fasta data dir. Should contain the sequences for the task.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the processed .tsv files are saved.",
    )

    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        help="The kmer used by the model",
    )


    args = parser.parse_args()

    extract_test_and_split_train_vali(args.data_dir, args.output_dir, args.kmer)



if __name__ == "__main__":
    main()
