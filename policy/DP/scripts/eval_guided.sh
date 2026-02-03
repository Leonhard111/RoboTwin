#  bash ./scripts/eval_guided.sh  3
gpu_id=${1}
export PYTHONPATH="$(pwd)":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${gpu_id}


python eval_test_time_optimization.py --config-name=eval_transport