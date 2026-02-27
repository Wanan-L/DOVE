# export PYTHONPATH=/data2/wujialing/miniconda3/envs/torch2.8/lib/python3.10/site-packages:$PYTHONPATH

python scripts/prepare_dataset.py --dir /data2/wujialing/code/VSR/DOVE/datasets/train/HQ-VSR

# train DOVE-s1
nohup bash train_ddp_one_s1.sh \
    >> logs/train_ddp_one_s1.log 2>&1 & tail -f logs/train_ddp_one_s1.log

cd /data2/wujialing/code/VSR/DOVE/finetune
bash train_ddp_one_s1.sh

# train DOVE-s2
python scripts/prepare_sft_ckpt.py --checkpoint_dir checkpoint/DOVE-s1/checkpoint-10000 \
    --weights_source /data2/wujialing/pretrain/Image-to-Video/CogVideoX1.5-5B

bash train_ddp_one_s2.sh

python scripts/prepare_sft_ckpt.py --checkpoint_dir checkpoint/DOVE-s2/checkpoint-500 \
    --weights_source /data2/wujialing/pretrain/Image-to-Video/CogVideoX1.5-5B

python scripts/prepare_sft_ckpt.py --checkpoint_dir checkpoint-10000/DOVE-s2/checkpoint-500 \
    --weights_source /data2/wujialing/pretrain/Image-to-Video/CogVideoX-2b