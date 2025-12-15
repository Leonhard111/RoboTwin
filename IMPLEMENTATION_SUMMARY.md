# collect_rollout.py 实现总结

## 实现完成情况

✅ 已成功完成所有需求功能：

### 1. 修改 main 函数 ✓
- **功能变更**：从评估策略改为收集数据
- **参数提取**：从配置文件中获取 `num_frames` 参数
- **执行流程**：
  - 加载配置和模型（同 eval_policy.py）
  - 调用 `collect_data()` 函数进行数据收集
  - 不再调用 `eval_policy()` 函数

### 2. 实现 collect_data 函数 ✓
基于 `eval_policy` 函数改写，但核心逻辑完全不同：

**参照的部分**：
- 环境设置流程
- 专家轨迹验证（expert_check）
- 缓存管理（clear_cache_freq）
- 策略执行循环结构

**新增的收集逻辑**：
- 收集头部相机 RGB 图像
- 收集关节状态向量
- 记录动作（下一个关节状态）
- 帧级别数据组织

### 3. 数据存储方式 ✓
完全按照 DP 的 `process_data.py` 格式：

**Zarr 存储结构**：
```
collected_data.zarr
├── data/
│   ├── head_camera: (N, 3, H, W)    # NCHW 格式图像
│   ├── state: (N, state_dim)         # 关节状态
│   └── action: (N, action_dim)       # 关节动作
└── meta/
    └── episode_ends: (M,)            # M 个回合的结束索引
```

**压缩设置**：
- 使用 Zstandard (zstd) 压缩
- 压缩级别：3
- 块大小：100 帧

### 4. 帧数控制 ✓
- **目标帧数**：通过 `num_frames` 参数指定
- **动态回合**：自动运行足够的回合直到达到目标
- **进度跟踪**：每 50 帧输出一次进度信息

### 5. 主函数只执行 collect_data ✓
- 移除了所有评估逻辑
- 移除了结果文件写入逻辑
- 清晰的单一职责：数据收集

## 关键代码特性

### 数据收集循环
```python
while total_frames < num_frames:
    # 验证专家轨迹
    # 设置环境和指令
    # 循环执行策略并收集数据
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and total_frames < num_frames:
        # 收集观测和状态
        # 执行策略动作
        # 记录下一状态
```

### 数据转换处理
- 图像：RGB -> BGR（使用 cv2）
- 格式转换：NHWC -> NCHW（numpy 数组轴变换）
- 数据类型：统一为 float32（便于 zarr 存储）

### 输出管理
生成的文件和目录：
- `collected_data.zarr`: 主数据文件
- `_collection_summary.txt`: 收集统计信息

## 配置文件

### collect_policy.yml
新建配置文件，继承 deploy_policy.yml 并添加：
- `num_frames`: 数据收集目标帧数（默认 100）

## 运行脚本

### collect_rollout.sh
便捷 shell 脚本，提供简单的命令行接口：
```bash
bash policy/DP/collect_rollout.sh <task> <config> <ckpt> <expert_num> <seed> <gpu> [num_frames]
```

## 文件清单

创建/修改的文件：
1. ✅ `/home/ubuntu/fjh/workbench/RoboTwin/script/collect_rollout.py` - 主脚本
2. ✅ `/home/ubuntu/fjh/workbench/RoboTwin/policy/DP/collect_policy.yml` - 配置文件
3. ✅ `/home/ubuntu/fjh/workbench/RoboTwin/policy/DP/collect_rollout.sh` - 运行脚本
4. ✅ `COLLECT_DATA_README.md` - 使用指南

## 与 eval_policy.py 的主要区别

| 功能 | eval_policy.py | collect_rollout.py |
|------|----------------|-------------------|
| 主要目的 | 评估策略成功率 | 收集策略数据 |
| 数据保存 | 统计文本 | Zarr 格式数据集 |
| 循环条件 | 固定 test_num | 动态 num_frames |
| 输出格式 | 成功率百分比 | 完整数据张量 |
| 数据粒度 | 回合级别 | 帧级别 |

## 使用示例

```bash
# 收集 500 帧数据
cd /home/ubuntu/fjh/workbench/RoboTwin
bash policy/DP/collect_rollout.sh beat_block_hammer beat_block_hammer_basic default 100 0 0 500

# 或直接调用 Python
python script/collect_rollout.py --config policy/DP/collect_policy.yml \
    --overrides \
    --task_name beat_block_hammer \
    --task_config beat_block_hammer_basic \
    --ckpt_setting default \
    --expert_data_num 100 \
    --seed 0 \
    --num_frames 500
```

## 验证

代码已通过以下检查：
- ✅ 无语法错误
- ✅ 所有导入完成
- ✅ 关键函数定义完整
- ✅ Zarr 数据格式正确
- ✅ 与 DP process_data.py 兼容
