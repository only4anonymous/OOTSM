export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 --master_port=12349 llama_SGA/SGA_stage_1.py \
  --data_path 'data/charades/ag' \
  --llama_path llama/Llama-3.2-3B-Instruct \
  --phase train \
  --datasize full \
  --context_fraction 0.9 \
  --epochs 3 \
  --batch_size 1 \
  --max_seq_length 2100 \
  --max_input_length 2100 \
  --lr 5e-6 \
  --lr_classify 5e-6 \
  --gradient_accumulation_steps 1 \
  --alpha 1 \
  --decode_length 256 \
  --transition_lambda 0.03 \
  --tau 0.2 \
  --temp_lambda 1.0 \
  --warmup_steps 1000 \
  --lora_r 32 \
  --lora_alpha 32 \
  --enable_classifier \
  --save_path llama_SGA/results/llama_3b_two_stage/stage_1_0.9_final_transition_loss_0.03 \
  --use_transition_loss \
