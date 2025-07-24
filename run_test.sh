
export CUDA_VISIBLE_DEVICES=1 
python test_sga_methods.py \
    --method_name=ode \
    --data_path="data/charades/ag" \
    --ckpt='checkpoints/multi_label/predcls_ode_3/ode_predcls_future_3/ode_predcls_future_3_epoch_9.tar' \
    --save_path="ckpt/llama_3b/fused_llm" \
    --results_path='results_llama_3b/fusion/0.9/multi_label/0.6_512' \
    --datasize mini \
    --llama_path="llama/Llama-3.2-3B-Instruct" \
    --lora_path="llama_SGA/results/llama_3b/0.9_prompt_simple/epoch_5" \
    --classifier_path="llama_SGA/results/llama_3b/0.9_prompt_simple/epoch_5/classifier.bin" \
    --model_path_stage0='llama/Llama-3.2-3B-Instruct' \
    --lora_path_stage0="llama_SGA/results/llama_3b_two_stage/weight/stage_0_0.5_lora_r_32_alpha_32_w0_label_debug_2/epoch_5" \
    --lora_path_stage1="llama_SGA/results/llama_3b_two_stage/stage_1_retrain_classify_head_4/epoch_5" \
    --use_llm \
    --three_stage \
    --threshold 0.6 \
    --stage1_tokens 512 \