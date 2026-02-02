
#  bash ./scripts/train_wm.sh  3
# pwd是当前命令行所在的目录
gpu_id=${1}
export PYTHONPATH="$(pwd)":$PYTHONPATH   #:是分隔符，后面是追加了原有值
export CUDA_VISIBLE_DEVICES=${gpu_id}

pwd
python dyn_model/train.py --config-name train.yaml 
