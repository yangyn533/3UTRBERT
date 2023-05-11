import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)
import math

from transformers import BertModel, RNATokenizer




def get_kmer_sentence(original_string, kmer=3):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string) - kmer):
        sequence.append(original_string[i:i + kmer])

    sequence.append(original_string[-kmer:])
    return sequence

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len (each layer_attention)
        print("layer_attention.shape: ", layer_attention.shape)
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0)) # layer_attention.squeeze(0), remove the 0th dimension if it is 1, num_heads x seq_len x seq_len
        
    # num_layers x num_heads x seq_len x seq_len

    return torch.stack(squeezed) #combine all metrics in squeezed to one 12 x 12 x seq_len x seq_len

def get_attention_3utr(model, tokenizer, sentence_a, start, end):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b=None, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    print("input_ids: ", input_ids)
    attention = model(input_ids)[-1]
    print("attention len: ", len(attention))
    print("attention shape: ", attention[-1].shape)
    input_id_list = input_ids[0].tolist() # Batch index 0
    #print("input_id_list: ", input_id_list)
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    print("tokens: ", tokens)
    attn = format_attention(attention)
    # print(attn)# num_layers x num_heads x seq_len x seq_len
    attn_score = []
    for i in range(1, len(tokens)-1):
        help = attn[start:end+1,:,0,i]
        # print(help)
        attn_score.append(float(attn[start:end+1,:,0,i].sum()))
    return attn_score

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


def generate_attention_average(args):

    if args.kmer == 0:
        KMER_LIST = [3, 4, 5, 6]

        for kmer in KMER_LIST:
            tokenizer_name = 'rna' + str(kmer)
            model_path = os.path.join(args.model_path, str(kmer))
            model = BertModel.from_pretrained(model_path, output_attentions=True)
            tokenizer = RNATokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
            raw_sentence = args.sequence
            sentence_a = ' '.join(get_kmer_sentence(raw_sentence, kmer))
            tokens = sentence_a.split()

            attention = get_attention_3utr(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer)
            attention_scores = np.array(attention).reshape(np.array(attention).shape[0],1)

            real_scores = get_real_score(attention_scores, kmer, args.metric)
            real_scores = real_scores / np.linalg.norm(real_scores)

            if kmer != KMER_LIST[0]:
                scores += real_scores.reshape(1, real_scores.shape[0])
            else:
                scores = real_scores.reshape(1, real_scores.shape[0])

    else:
        model_path = args.model_path
        model = BertModel.from_pretrained(model_path, output_attentions=True)
        tokenizer = RNATokenizer.from_pretrained(model_path, do_lower_case=False)
        raw_sentence = args.sequence
        sentence_a = ' '.join(get_kmer_sentence(raw_sentence, args.kmer))
        tokens = sentence_a.split()
        attention = get_attention_3utr(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer)
        attention_scores = np.array(attention).reshape(np.array(attention).shape[0],1)
        real_scores = get_real_score(attention_scores, args.kmer, args.metric)
        scores = real_scores.reshape(1, real_scores.shape[0])

    ave = np.sum(scores)/scores.shape[1]

    return real_scores


def highlighter(i, weight, input_seq_list):
    help = weight[i]
    max_item = math.ceil(max(weight))
    color = '#%02X%02X%02X' % (
        255, int(255*(1-weight[i]/max_item)), int(255*(1-weight[i]/max_item)))
    word = '<span style="background-color:' +color+ '">' +input_seq_list[i]+ '</span>'
    return word





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--model_path",
        default="",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--start_layer",
        default=11,
        type=int,
        help="Which layer to start",
    )
    parser.add_argument(
        "--end_layer",
        default=11,
        type=int,
        help="which layer to end",
    )
    parser.add_argument(
        "--metric",
        default="mean",
        type=str,
        help="the metric used for integrate predicted kmer result to real result",
    )
    parser.add_argument(
        "--sequence",
        default="",
        type=str,
        help="the sequence for visualize",
    )
    parser.add_argument(
        "--save_path",
        default="",
        type=str,
        help="the directory for output",
    )

    args = parser.parse_args()
    weight = generate_attention_average(args)
    text = ''.join([highlighter(i, weight, args.sequence) for i in range(len(args.sequence))])
    with open(args.save_path + "/single_resolution_importance.html", "w") as file:
        file.write(text)
    return weight.tolist()

if __name__ == "__main__":
    attention = main()
    print(attention)

    
    
    
