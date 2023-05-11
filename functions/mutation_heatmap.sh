export KMER=3
export MODEL_PATH=<PATH_TO_YOUR_MODEL>
export ORIGINAL_SEQ_PATH=<PATH_TO_YOUR_ORIGINAL_SEQUENCE_FILE>
export MUTATE_SEQ_PATH=<PATH_TO_YOUR_MUTATED_SEQUENCE_FILE>
export PREDICTION_PATH=<PATH_TO_STORE_PREDICTION>
export WT_SEQ=<THE_SEQUENCE_USED_FOR_MUTATION>
export OUTPUT_PATH=<PATH_TO_YOUR_OUTPUT_DIRECTORY>


# mutate sequence
python mutate_seqs.py --seq_file $ORIGINAL_SEQ_PATH/test.tsv --save_file_dir $MUTATE_SEQ_PATH --k $KMER


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
