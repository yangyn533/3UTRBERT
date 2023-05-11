#### ::: DNABERT-viz SNP analysis ::: ####

import os
import sys
sys.path.append('../motif')
import pandas as pd
import numpy as np
import argparse
#import motif_utils as utils

def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    
    Arguments:
    kmers -- str, kmers separated by space.
    
    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    print(kmers_list)
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    print(bases)
    seq = "".join(bases)
    print(len(seq))
    print(seq)
    print(len(kmers_list))
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_seq_file",
        default='../examples/sample_data/ft/prom-core/6/dev.tsv',
        type=str,
        required=True,
        help="Path to original input sequence+label .tsv file.",
    )
    
    parser.add_argument(
        "--orig_pred_file",
        required=True,
        type=str,
        default='../examples/result/prom-core/6/pred.npy',
        help="Path to predictions pred.npy of original sequences.",
    )
    
    parser.add_argument(
        "--mut_seq_file",
        default='examples/dev.tsv',
        type=str,
        required=True,
        help="Path to mutated sequence+index .tsv file.",
    )
    
    parser.add_argument(
        "--mut_pred_file",
        required=True,
        type=str,
        default='examples/pred.npy',
        help="Path to predictions pred_results.npy of mutated sequences.",
    )
    
    parser.add_argument(
        "--save_file_dir",
        default='.',
        type=str,
        help="Path to save outputs",
    )

    # TODO: add the conditions
    args = parser.parse_args()

    # original sequences
    # orig_pred = np.load(args.orig_pred_file)
    orig_dev = pd.read_csv(args.orig_seq_file,sep='\t',header=0)
    orig_dev.columns = ['sequence','label']
    orig_dev['orig_seq'] = orig_dev['sequence'].apply(kmer2seq)
    orig_dev['idx'] = orig_dev.index
    
    orig_pred = np.load(args.orig_pred_file)
    orig_dev['orig_pred'] = orig_pred
    
    # mutated sequences
    # mut_pred = np.load(args.mut_pred_file)
    mut_dev = pd.read_csv(args.mut_seq_file,sep='\t',header=0)
    mut_dev.columns = ['sequence','label','idx'] #ignore label
    mut_dev['mut_seq'] = mut_dev['sequence'].apply(kmer2seq)
    
    mut_pred = np.load(args.mut_pred_file)
    mut_dev['mut_pred'] = mut_pred
    
    # merge
    dev = pd.merge(orig_dev[['idx','orig_seq','orig_pred']],
                        mut_dev[['idx','mut_seq','mut_pred']],
                        on='idx'
                       )
    dev['diff'] = (dev['mut_pred'] - dev['orig_pred'])*(dev[['orig_pred','mut_pred']].max(axis=1))
    dev['logOR'] = np.log2(dev['orig_pred']/(1-dev['orig_pred'])) - np.log2(dev['mut_pred']/(1-dev['mut_pred']))
    dev.to_csv(os.path.join(args.save_file_dir,'mutations.tsv'),sep='\t')
    
if __name__ == "__main__":
    main()
