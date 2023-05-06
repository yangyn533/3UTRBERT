# 3UTRBERT
## Environment Setup

#### 1.1 Create and activate a new virtual environment
```
conda create -n 3UTRBERT python=3.6.13 
conda activate 3UTRBERT
```
#### 1.2 Install the package and other requirements
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
git clone https://github.com/yangyn533/3UTRBERT
cd 3UTRBERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```
## Process data
The input file is in .fasta format. For each sequence, the label of the sequence should be in the sequence ID. (example file can be dound in example_data folder).
By running the following code, the input fasta file will be separated into train, dev and test sets. Each sequence will be tokenized into 3mer tokens.
```
python preprocess.py \
  --data_dir <PATH_TO_YOUR_DATA> \
  --output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --kmer 3
```
## Train
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
## Predict
```
python predict.py \
--data_dir <PATH_TO_YOUR_DATA> \
--output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
--do_predict \
--tokenizer_name rna3 \
--model_type 3utrprom \
--model_name_or_path <PATH_TO_YOUR_MODEL> \
--max_seg_length 100 \
--per_gpu_eval_batch_size 32
```
Please change the tokenizer name { rna3, rna4, rna5, rna6 } when changing the kmer choice.
## Single resolution importance analysis
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

## Motif analysis
CHECK IF THIS IS NECESSARY
```
python find_motifs.py \
    --data_dir <PATH_TO_YOUR_DATA> \
    --predict_dir <PATH_TO_YOUR_PREDICTION_OUTPUT_DIRECTORY> \
    --window_size 7 \
    --min_len 5 \
    --pval_cutoff 0.05 \
    --min_n_motif 1 \
    --align_all_ties \
    --save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
    --verbose
```

## Mutation
```
python mutate.py \
--seq_file <PATH_TO_SEQUENCE_FILE>\
--save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
--k 3
```
```
python predict.py \
    --model_type 3utrprom \
    --tokenizer_name rna3 \
    --model_name_or_path  <PATH_TO_YOUR_MODEL>\
    --task_name rnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 100 \
    --per_gpu_pred_batch_size=128   \
    --output_dir <PATH_TO_YOUR_MODEL> \
    --predict_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
    --n_process 48
```

```
python calculate_diff_scores.py \
--orig_seq_file <PATH_TO_YOUR_OTIGINAL_SEQ_FILE> \
--orig_pred_file <PATH_TO_YOUR_OTIGINAL_SEQ_PREDICTION> \
--mut_seq_file <PATH_TO_YOUR_MUTATED_SEQ_FILE> \
--mut_pred_file <PATH_TO_YOUR_MUTATED_SEQ_PREDICTION> \
--save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY>
```
remember to add the heatmap drawing codes

## Feature extraction
```
remember to add the python file
```
