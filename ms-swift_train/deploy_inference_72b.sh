cd /data/LLM/wjj/模型推理
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift deploy \
    --adapters /data/LLM/WSC/RLHF/output/v3-20250911-183715/checkpoint-1500 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 8192 \
    --vllm_tensor_parallel_size 4 \
    --max_new_tokens 1024 \
    --host 0.0.0.0 \
    --port 8088 \