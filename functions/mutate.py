import os
import sys
import pandas as pd
import numpy as np
import argparse
# import motif_utils as utils

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

def mutate(seq, start, end, target=None):
    """
    Mutate input sequence at specified position.
    
    If target is not None, returns the mutated seq. Otherwise, returns a numpy array with shape (4,1)
    with all four mutated possibilities.
    
    Arguments:
    seq -- str, original sequence.
    start -- int, starting index where nucleotide needs to be changed. Counting starts at zero.
    end -- int, ending index where nucleotide needs to be changed. Counting starts at zero.
    
    Keyword arguments:
    target -- str, the target nucleotide(s) to be changed to (default: None).
    
    Returns:
    mutated_seq -- str, mutated sequence.

    """
    assert end >= start and start >= 0 and end <= len(seq), "Wrong start and end index input."
    
    if target is not None:
        mutated_seq = seq[:start] + str(target) + seq[end:]
    else:
        mutated_seq = []
        for n in ['A','U','G','C']:
            m_seq = seq[:start] + str(n) + seq[end:]
            mutated_seq.append(m_seq)
        mutated_seq = np.asarray(mutated_seq)
    return mutated_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seq_file",
        type=str,
        help="Path to input sequence + label .tsv file.",
    )
    
    parser.add_argument(
        "save_file_dir",
        type=str,
        help="Path to save the mutated seqs",
    )
    
    parser.add_argument(
        "--mut_file",
        default=None,
        type=str,
        help="Path to the file defining how each input seq should be mutated",
    )
    
    parser.add_argument(
        "--k",
        default=3,
        type=int,
        help="length of kmer for conversion of mutated seqs"
    )

    
    args = parser.parse_args()
    
    os.makedirs(args.save_file_dir, exist_ok=True)
    
    mutated_dev = {'index':[],'seq':[]}
    
    dev = pd.read_csv(args.seq_file,sep='\t',header=0)
    dev.columns = ['sequence','label']
    dev['seq'] = dev['sequence'].apply(kmer2seq)
    
    if args.mut_file is not None:
        mut_file = pd.read_csv(args.mut_file, sep='\t',header=None)
        mut_file = mut_file.fillna('')
        mut_file.columns = ['idx','start', 'end', 'allele']
        mut_file['idx'] = mut_file['idx'].astype(int)
        mut_file['start'] = mut_file['start'].astype(int)
        mut_file['end'] = mut_file['end'].astype(int)
        dev_selected = dev.iloc[mut_file['idx'].tolist(),:].reset_index()
        for i, row in dev_selected.iterrows():
            seq = row['seq']
            mut = mut_file.iloc[i]
            mut_seq = mutate(seq, mut['start'], mut['end'], target = mut['allele'])
            mut_seq = seq2kmer(mut_seq, args.k)
            mutated_dev['index'].append(mut['idx'])
            mutated_dev['seq'].append(mut_seq)
    else:
        for i, row in dev.iterrows():
            seq = row['seq']
            for j in range(len(seq)):
                mut_seq = mutate(seq, j, j+1)
                mut_seq = [seq2kmer(seq, args.k) for seq in mut_seq]
                idx = [i] * 4
                mutated_dev['index'].extend(idx)
                mutated_dev['seq'].extend(mut_seq)

    mutated_dev = pd.DataFrame.from_dict(mutated_dev)
    mutated_dev = mutated_dev[['seq','index']]
    mutated_dev.columns = ['sequence','index']
    mutated_dev['label'] = 0
    mutated_dev.iloc[0, mutated_dev.columns.get_loc('label')] = 1
    mutated_dev = mutated_dev[['sequence','label','index']]
            
    mutated_dev.to_csv(os.path.join(args.save_file_dir,'dev.tsv'),sep='\t',header=True, index=False)
    

if __name__ == "__main__":
    main()