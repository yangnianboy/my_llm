pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
swanlab login -k qY96Dsu2lopKY7C4h6Ho6
modelscope download --dataset gongjy/minimind_dataset prtrain_hq.jsonl sft_512.jsonl sft_2048.jsonl dpo.jsonl --cache_dir ../dataset
torchrun --nproc_per_node 8 trainer/train_pretrain.py --epochs 3 --batch_size 128 --num_workers 8 --accumulation_steps 2 --hidden_size 768 --num_hidden_layers 16 --wandb_project gugugaga-Pretrain-large
torchrun --nproc_per_node 8 trainer/train_full_sft.py --epochs 2 --batch_size 128 --num_workers 8 --accumulation_steps 2 --hidden_size 768 --num_hidden_layers 16 --wandb_project gugugaga-SFT-large --max_seq_len 350 --data_path ../datasets/sft_512.jsonl --wandb_project gugugaga-SFT-large-514
torchrun --nproc_per_node 8 trainer/train_full_sft.py --epochs 2 --batch_size 128 --num_workers 8 --accumulation_steps 2 --hidden_size 768 --num_hidden_layers 16 --wandb_project gugugaga-SFT-large --max_seq_len 1400 --data_path ../datasets/sft_2048.jsonl --wandb_project gugugaga-SFT-large-2048
torchrun --nproc_per_node 8 trainer/train_dpo.py --epochs 2 --batch_size 128 --num_workers 8 --accumulation_steps 2 --hidden_size 768 --num_hidden_layers 16 --wandb_project gugugaga-DPO-large
python eval_llm.py --weight dpo
