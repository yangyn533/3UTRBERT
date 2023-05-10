
export KMER=3
export MODEL_PATH=<>
export ORIGINAL_SEQ_PATH=<>
export PREDICTION_PATH=<>
export MUTATE_SEQ_PATH=<>
export WT_SEQ=<>


# mutate sequence
python mutate_seqs.py --seq_file <PATH_TO_TSV_FILE> --save_file_dir <PATH_TO_OUTPUT_DIRECTORY> --k <KMER_USED>


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
  --orig_seq_file  $ORIGINAL_SEQ_PATH \
  --orig_pred_file  $PREDICTION_PATH/original_pred/pred_results.npy \
  --mut_seq_file  $MUTATE_SEQ_PATH \
  --mut_pred_file $PREDICTION_PATH/mutate_pred/pred_results.npy \
  --save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY>


# draw heatmap
python heatmap.py \
  --score_file <PATH_TO_SCORE_TSV_FILE> \
  --save_file_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> \
  --wt_seq $WT_SEQ