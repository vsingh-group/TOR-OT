#!/bin/bash
source /home/ivan/anaconda3/envs/env_TF4OT/bin/activate env_TF4OT || true

METRICS=("sqeuclidean" "euclidean" "cityblock" "chebyshev" "canberra" "minkowski")

for metric in "${METRICS[@]}"
do
    echo "=================================================="
    echo "Running inference for metric: $metric"
    echo "=================================================="
    
    PYTHONPATH=. /home/ivan/anaconda3/envs/env_TF4OT/bin/python TF4OT/run_inference_attention_visual.py \
      --config-name TF_100K_cost_semivariable.yaml \
      data.dataset.size_s=\"600K\" \
      data.dataset.size_n=600000 \
      tokenizer.DigitTokenizer.decimal_precision=4 \
      data.dataset.input_cost_metric=$metric \
      inference.test_over_testset=false \
      inference.test_over_custom_pairs=true
done

echo "Evaluation complete across all metrics."
