export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=12344 llama_SGA/SGA_stage_0_data.py \
  --data_path 'data/charades/ag' \
  --llama_path llama/Llama-3.2-3B-Instruct \
  --phase train \
  --datasize full \
  --context_fraction 0.7 \
  --epochs 5 \
  --batch_size 1 \
  --max_seq_length 1800 \
  --lr 1e-5 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 1000 \
  --lora_r 32 \
  --lora_alpha 32 \
  --beta 0.3 \
  --save_path llama_SGA/results/llama_3b_two_stage/weight/stage_0_final/0.7_len1_1_0.3\
  --len 1 \