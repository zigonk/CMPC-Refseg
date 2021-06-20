#!/usr/bin/env bash

LOG=logs/cmpc_model
mkdir -p ${LOG}
now=$(date +"%Y%m%d_%H%M%S")

python -u trainval_model.py \
-m train \
-d refvos \
-bs 8 \
-datadir /home/zigonk/Documents/Thesis/CMPC-Refseg/data_preprocessing \
-t train \
-n CMPC_model \
-i 100000 \
-finetune \
-emb \
-embdir /home/zigonk/Documents/Thesis/CMPC-Refseg/data \
-f /home/zigonk/Documents/Thesis/CMPC-Refseg/model/

python -u trainval_model.py \
-m test \
-d unc \
-t val \
-n CMPC_model \
-i 700000 \
-c \
-emb \
-f ckpts/unc/cmpc_model 2>&1 | tee ${LOG}/test_val_$now.txt
