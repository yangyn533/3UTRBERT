# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import argparse
import glob
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import re
import shutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from multiprocessing import Pool
from typing import Dict, List, Tuple

#import neptune.new as neptune
import numpy as np
import torch
from decouple import config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from data_loaders import load_and_cache_examples_3utr as load_and_cache_examples
from data_loaders import visualize


from src.transformers import (
    RNATokenizer
)

from transformers import (
    BertConfig,
    DistilBertConfig,
    WEIGHTS_NAME,
    AdamW,
    BertForSequenceClassification,
    DistilBertForSequenceClassification
)

from src.transformers import glue_compute_metrics as compute_metrics
from src.transformers import glue_output_modes as output_modes
from src.transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "3utrprom": (BertConfig, BertForSequenceClassification, RNATokenizer),
    "distil3utrprom": (DistilBertConfig, DistilBertForSequenceClassification, RNATokenizer),
    "mini3utrprom": (BertConfig, BertForSequenceClassification, RNATokenizer),
}

TOKEN_ID_GROUP = ["bert", "3utrlong", "3utrlongcat", "xlnet", "albert"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)




def predict(args, model, tokenizer, prefix=""):
    pred_task = args.task_name
    pred_output_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    softmax = torch.nn.Softmax(dim=1)

    predictions = {}
    pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.pred_batch_size)
    pred_loss = 0.0
    nb_pred_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(pred_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(pred_task, preds, out_label_ids, probs)
    print('*****result*****', result)
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedir(pred_output_dir)
    output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")
    logger.info("***** Pred results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    np.save(output_pred_file, result)



def main():
    parser = argparse.ArgumentParser()

    # BASIC
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list",)
    parser.add_argument("--task_name", default='rnaprom', type=str, help="Script only prepared for promoter task" )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--tokenizer_name",default="rna3",type=str, help="Pretrained tokenizer name or path if not the same as model_name",)

    # OBJECTIVE
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")

    # VALIDATION DURING TRAINING / EVALUATE
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",)
    parser.add_argument("--do_visualize_during_training", action="store_true", help="Steps to generate an image")
    parser.add_argument("--image_steps", type=int, default=0, help="Steps to generate an image")

    # MODEL CONFIGS (only use)

    # TRAINING DETAILS
    parser.add_argument("--max_seq_length", default=384, type=int, help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument("--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--n_process", default=2, type=int, help="number of processes used for data process",)
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",)
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--neptune", default=False, help="Neptune")
    parser.add_argument("--neptune_tags", type=list, default=["trial"], help="Neptune tags")
    parser.add_argument("--neptune_description", type=str, default="TRIAL minilm fine-tuning", help="Neptune description")
    parser.add_argument("--neptune_token", type=str, default=None, help="Neptune API token")
    parser.add_argument("--neptune_project", type=str, default=None, help="Neptune project")



    # OTHER
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)


    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    # Segons els que he entès, local_rank és per si utilitzes més d'una màquina. Pot ser que utilitzis 1+ gpu però totes a la mateixa màquina
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print("devices", device)
        print("Number of gpus", args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]  # 'classification' or 'regressions'
    label_list = processor.get_labels()  # [0,1]
    num_labels = len(label_list)


    # LOAD AND INITIALIZE MODELS -----------------------------------------------------------------------------------------------------

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


    # PREDICTION OF THE TEST SET + EVALUATE -----------------------------------------------------------------------------------------
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)
        prediction = predict(args, model, tokenizer)



if __name__ == "__main__":
    main()
