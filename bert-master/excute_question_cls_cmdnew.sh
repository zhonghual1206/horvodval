#!/bin/bash

#source activate tensorflow

now=$(date +"%m%d")
log_file=bert${now}_deal_param.log


output_dir=./deal_param_output_zhl
mkdir -p ${output_dir}

export BERT_BASE_DIR=./chinese_L-12_H-768_A-12

horovodrun -np 2 -H 192.168.99.118:1,192.168.99.118:1 -p 2376
	python -u run_classifier.py \
    --task_name=deal  \
    --do_train=true \
    --do_eval=false \
    --do_predict=false  \
    --data_dir=./data_dir/  \
    --vocab_file=$BERT_BASE_DIR/vocab.txt  \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt  \
    --max_seq_length=64 \
    --train_batch_size=10 \
    --learning_rate=5e-5 \
    --num_train_epochs=50.0 \
    --output_dir=${output_dir} 1>${log_file} 2>&1 &


