#!/bin/bash

# CH SBA
python tools/aht_scores_eval.py \
    --aht_model ch_e50_sba_sad \
    --train_partner_model sad \
    --partner_model sad_aht \
    --split_type six \
    --split_indices 0-1 \
