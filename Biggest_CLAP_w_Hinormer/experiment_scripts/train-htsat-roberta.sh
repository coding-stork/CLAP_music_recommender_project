#!/bin/bash

# sent to sub script
echo go $COUNT_NODE
echo $HOSTNAMES

export PYTHONPATH=/content/CLAP/src/laion_clap:$PYTHONPATH

cd /content/CLAP/src
export TRANSFORMERS_CACHE=/content/transformers_cache

torchrun --nproc_per_node=1 laion_clap/training/main.py \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --batch-size=48 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=300 \
    --workers=1 \
    --use-bn-sync \
    --amodel HTSAT-base \
    --tmodel roberta \
    --warmup 1600 \
    --report-to "wandb" \
    --wandb-notes "10.16-clap-dataset-1#-htsat-roberta" \
    --datasetnames "smaller_spotifytrain" "audiocaps" \
    --datasetinfos "train" "unbalanced_train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir /content/transformers_cache \
    --seed 3407 \
    --lensequence 50 \
    --datasetpath '/content' \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --h_len-seq 250 \
    --h_dropout 0.5 \
    --h_beta 0.5 \
    --h_temperature 2 \
    --h_num-layers 4 \
    --h_num-gnns 2 \
    --h_num-heads 4 \
    --hinormer_weights_path "/content/HINormer_spotify_4_0.pt" \
    --pretrained-audio /content/music_audioset_epoch_15_esc_90.14.pt