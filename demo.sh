#!/bin/bash

vessel_types=("RCA" "LCX" "LAD")

for vessel in "${vessel_types[@]}"; do
    python ./tube_generator.py \
        --save_path="generated" \
        --dataset_name="test" \
        --num_trees=1 \
        --num_branches=3 \
        --vessel_type="$vessel" \
        --shear \
        --save_visualization \
        --generate_projections &
done
