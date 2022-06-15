#! /bin/bash

# Autoformer
run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_720 --model Autoformer --batch_size 12 --train_epochs 3 --features SA --seq_len 720 --pred_len 720 --e_layers 2 --d_layers 1 --d_model 512 --d_ff 2048 --des 'Exp'
# Informer
run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_720 --model Informer --batch_size 12 --train_epochs 3 --features SA --seq_len 720 --pred_len 720 --e_layers 2 --d_layers 1 --d_model 512 --d_ff 2048 --des 'Exp'
# Reformer
run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_720 --model Reformer --batch_size 12 --train_epochs 3 --features SA --seq_len 720 --pred_len 720 --e_layers 2 --d_layers 1 --d_model 512 --d_ff 2048 --des 'Exp'
# Transformer
run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_720 --model Transformer --batch_size 12 --train_epochs 3 --features SA --seq_len 720 --pred_len 720 --e_layers 2 --d_layers 1 --d_model 512 --d_ff 2048 --des 'Exp'
