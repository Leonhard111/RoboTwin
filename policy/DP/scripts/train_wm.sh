
#  bash ./scripts/train_wm.sh  3
# pwd是当前命令行所在的目录
# PS: hydra 好像是先 默认配置，后override，最后计算${}引用
gpu_id=${1}
export PYTHONPATH="$(pwd)":$PYTHONPATH   #:是分隔符，后面是追加了原有值
export CUDA_VISIBLE_DEVICES=${gpu_id}
task_name='hanging_mug'

pwd
python dyn_model/train.py --config-name train.yaml \
            env.train_data_path="/home/ubuntu/fjh/workbench/RoboTwin/policy/DP/data/${task_name}-demo_clean-50.zarr" \
            env.val_data_path="/home/ubuntu/fjh/workbench/RoboTwin/policy/DP/data/${task_name}-demo_clean-50.zarr" \
            env.policy_ckpt_path="/home/ubuntu/fjh/workbench/RoboTwin/policy/DP/data/outputs/${task_name}-maniflow_image_timm_policy_robotwin2-1001_seed0/checkpoints/latest.ckpt"
