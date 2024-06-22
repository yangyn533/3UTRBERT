
# ![image](https://github.com/yangyn533/3UTRBERT/blob/main/3UTRBERT.png)
## ⌛️ Data availability
[3UTRBERT_dataset](https://doi.org/10.6084/m9.figshare.26082916.v1)

## ⌛️ Download pre-trained 3UTRBERT model
[3UTRBERT-3mer](https://figshare.com/articles/software/Pre-trained_3mer_model/22847354)

[3UTRBERT-4mer](https://figshare.com/articles/software/Pre-trained_4mer_model/22851119)

[3UTRBERT-5mer](https://figshare.com/articles/software/Pre-trained_5mer_model/22851191)

[3UTRBERT-6mer](https://figshare.com/articles/software/Pre-trained_6mer_model/22851272)

## 📘 Environment Setup

#### 1.1 Create and activate a new virtual environment
```
conda create -n 3UTRBERT python=3.6.13 
conda activate 3UTRBERT
```

#### 1.2 Install the package and other requirements
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
git clone https://github.com/yangyn533/3UTRBERT
cd 3UTRBERT
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt
```

If above commands do not run correctly. Following commands could be used to install missing packages manually **after running the above commands.** If you use the commands to install packages manually, above commands should be run first.

```
pip install seaborn
pip install transformers
pip install pyfaidx
pip install python-decouple
pip install sacremoses
pip install boto3
pip install sentencepiece
pip install Bio
pip install pyahocorasick
```


## ⌛️ Process data
The input file is in `.fasta` format. For each sequence, the label of the sequence should be in the sequence ID. (example file can be found in example_data folder).
By running the following code, the input fasta file will be separated into train, dev and test sets. Each sequence will be tokenized into 3mer tokens. Example data locates in the example/data folder. `train.tsv` is for training, `dev.tsv` for validation and `test.tsv` for test the performance.
```
python preprocess.py \
  --data_dir <PATH_TO_YOUR_DATA> \
  --output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --kmer 3
```


## ⌛️ Train
`train.py` is used for fine-tune the model. The input data are `train.tsv` and `dev.tsv`. Make sure `train.tsv` and `dev.tsv` are in the same directory and the input path to this directory as the `--data_dir` argument (not include the file name itself). `--model_name_or_path` needs to be the path to your pre-trained model. `--output_dir` is the location to store the fine-tuned model.
```
python train.py \
  --data_dir <PATH_TO_YOUR_DATA> \
  --output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --model_type 3utrprom \
  --tokenizer_name rna3 \
  --model_name_or_path <PATH_TO_YOUR_MODEL> \
  --do_train \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --logging_steps 100 \
  --save_steps 1000 \
  --num_train_epochs 3 \
  --evaluate_during_training \
  --max_seq_length 100 \
  --warmup_percent 0.1 \
  --hidden_dropout_prob 0.1 \
  --overwrite_output \
  --weight_decay 0.01 \
  --seed 6
```
Please change the tokenizer name { rna3, rna4, rna5, rna6 } when changing the kmer choice.


## ⌛️ Predict
`predict.py` is used for producing prediction results from the fine-tuned model. The input data is the `test.tsv`. Make sure `train.tsv`, `dev.tsv` and `test.tsv` are in the same directory and input path to this directory as the `--data_dir` argument (not include the file name itself). `--model_name_or_path` needs to be the path to your fine-tuned model. The output files of `predict.py` are mainly `pred_results.npy` and `pred_results_scores.npy`. `pred_results.npy` stores the probability for each sequence. `pred_results_scores.npy` stores the metrics to evaluate the model.
```
python predict.py \
  --data_dir <PATH_TO_YOUR_DATA> \
  --output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --do_predict \
  --tokenizer_name rna3 \
  --model_type 3utrprom \
  --model_name_or_path <PATH_TO_YOUR_MODEL> \
  --max_seq_length 100 \
  --per_gpu_eval_batch_size 32
```
Please change the tokenizer name { rna3, rna4, rna5, rna6 } when changing the kmer choice.


## 📊 Single resolution importance analysis
The following code extracted the attention scores and visualizes them.
```
python single_resolution_importance.py \
    --kmer 3 \
    --model_path <PATH_TO_YOUR_MODEL> \
    --start_layer 11 \
    --end_layer 11 \
    --metric mean \
    --sequence <SEQUENCE_USED> \
    --save_path <PATH_TO_YOUR_OUTPUT_DIRECTORY>
```
Please make sure that the input sequence does not exceed the max-length limit.


## 📊 Mutation analysis
Before run the shell script. Make sure the parameters in the shell script are indicated.
```
source mutation_heatmap.sh
```
The following commonds comes from `mutation_heatmap.sh`.
`KMER` indicates the kmer used. 
`ORIGINAL_SEQ_PATH` should be the path the the folder where your sequence file locates (not include the file name itself).
`MUTATE_SEQ_PATH` should be the folder your want to store the mutated sequence file (not include the file name itself). 
`WT_SEQ` should be the same sequence in the original sequence file.
Please store the original sequence in a `.tsv` file called `test.tsv`.

The example `test.tsv` file is in example `mutation_analysis/original_seq`, only one sequence is allowed to be in the file. The label for the sequence can be 0 or 1.
```
export KMER=3
export MODEL_PATH=<PATH_TO_YOUR_MODEL>
export ORIGINAL_SEQ_PATH=<PATH_TO_YOUR_ORIGINAL_SEQUENCE_FILE>
export MUTATE_SEQ_PATH=<PATH_TO_YOUR_MUTATED_SEQUENCE_FILE>
export PREDICTION_PATH=<PATH_TO_STORE_PREDICTION>
export WT_SEQ=<THE_SEQUENCE_USED_FOR_MUTATION>
export OUTPUT_PATH=<PATH_TO_YOUR_OUTPUT_DIRECTORY>


# mutate sequence
python mutate.py --seq_file $ORIGINAL_SEQ_PATH/test.tsv --save_file_dir $MUTATE_SEQ_PATH --k $KMER


# predict on sequence
mkdir $PREDICTION_PATH/original_pred
mkdir $PREDICTION_PATH/mutate_pred

python predict.py \
  --data_dir $ORIGINAL_SEQ_PATH \
  --output_dir $PREDICTION_PATH/original_pred \
  --do_predict \
  --tokenizer_name rna3 \
  --model_type 3utrprom \
  --model_name_or_path $MODEL_PATH \
  --max_seg_length 100 \
  --per_gpu_eval_batch_size 32

python predict.py \
  --data_dir $MUTATE_SEQ_PATH \
  --output_dir $PREDICTION_PATH/mutate_pred \
  --do_predict \
  --tokenizer_name rna3 \
  --model_type 3utrprom \
  --model_name_or_path $MODEL_PATH \
  --max_seg_length 100 \
  --per_gpu_eval_batch_size 32


# calculate scores
python calculate_diff_scores.py \
  --orig_seq_file  $ORIGINAL_SEQ_PATH/test.tsv \
  --orig_pred_file  $PREDICTION_PATH/original_pred/pred_results.npy \
  --mut_seq_file  $MUTATE_SEQ_PATH/test.tsv \
  --mut_pred_file $PREDICTION_PATH/mutate_pred/pred_results.npy \
  --save_file_dir $OUTPUT_PATH


# draw heatmap
python heatmap.py \
  --score_file $OUTPUT_PATH \
  --save_file_dir $OUTPUT_PATH \
  --wt_seq $WT_SEQ
```


## 📊 Feature extraction
`<PATH_TO_DATA>` is the path the folder that the data in (not include the data file name). The input fasta file should be named as `seq_to_extract.fasta`. THe example data can be found in example folder.
```
python extract_LS_embedding.py \
    --data_path <PATH_TO_DATA> \
    --output_path <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
    --model_path <PATH_TO_YOUR_MODEL>
```


## 🧬 Motif analysis
The motif analysis requires the output of attentions. The required attention can be obtained from `single_resolution_importance.py`. Store the attention into the directory used as input `--predict_dir`.
```
python find_motifs.py \
    --data_dir <PATH_TO_YOUR_DATA> \
    --predict_dir <PATH_TO_YOUR_PREDICTION_OUTPUT_DIRECTORY> \
    --window_size <ADJUST_THIS> \
    --min_len <ADJUST_THIS> \
    --pval_cutoff <ADJUST_THIS> \
    --min_n_motif <ADJUST_THIS> \
    --align_all_ties \
    --save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
    --verbose
```
