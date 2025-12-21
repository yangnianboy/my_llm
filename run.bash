#!/bin/bash
set -e

# --- Config ---
NPROC=1
HIDDEN=512
LAYERS=8
BATCH=64
ACC=4
WORKERS=16
DATA_DIR="./dataset"

# --- Training ---
echo ">>> 1. Pretrain"
torchrun --nproc_per_node $NPROC trainer/train_pretrain.py \
    --epochs 3 --batch_size $BATCH --num_workers $WORKERS --accumulation_steps $ACC \
    --hidden_size $HIDDEN --num_hidden_layers $LAYERS \
    --wandb_project gugugaga-Pretrain-large

echo ">>> 2. SFT (Seq 350)"
torchrun --nproc_per_node $NPROC trainer/train_full_sft.py \
    --epochs 2 --batch_size $BATCH --num_workers $WORKERS --accumulation_steps $ACC \
    --hidden_size $HIDDEN --num_hidden_layers $LAYERS \
    --max_seq_len 350 --data_path "$DATA_DIR/sft_512.jsonl" \
    --wandb_project gugugaga-SFT-large-514

echo ">>> 3. SFT (Seq 1400)"
torchrun --nproc_per_node $NPROC trainer/train_full_sft.py \
    --epochs 2 --batch_size $BATCH --num_workers $WORKERS --accumulation_steps $ACC \
    --hidden_size $HIDDEN --num_hidden_layers $LAYERS \
    --max_seq_len 1400 --data_path "$DATA_DIR/sft_2048.jsonl" \
    --wandb_project gugugaga-SFT-large-2048

echo ">>> 4. DPO"
torchrun --nproc_per_node $NPROC trainer/train_dpo.py \
    --epochs 2 --batch_size $BATCH --num_workers $WORKERS --accumulation_steps $ACC \
    --hidden_size $HIDDEN --num_hidden_layers $LAYERS \
    --wandb_project gugugaga-DPO-large

echo ">>> Evaluation"
python eval_llm.py --weight dpo --hidden_size  768 --num_hidden_layers 16 --lora_weight lora_identity
python api_server.py --weight dpo --hidden_size  768 --num_hidden_layers 16 --lora_weight lora_identity
