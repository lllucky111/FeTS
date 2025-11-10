model_name=FeTS

root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2021


python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --d_model 128 \
  --head_dropout 0.1 \
  --des Exp \
  --lradj type3 \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 256 \
  --patience 20 \
  --learning_rate 0.0001 \
  --random_seed $random_seed

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 7 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --d_model 128 \
  --head_dropout 0.1 \
  --dropout 0.5 \
  --des Exp \
  --lradj type3 \
  --itr 1 \
  --train_epochs 50 \
  --batch_size 256 \
  --patience 20 \
  --learning_rate 0.0001 \
  --random_seed $random_seed
