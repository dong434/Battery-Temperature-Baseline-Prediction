import os

device = 'cpu'
lr = 0.001
weight_decay = 5e-5
epochs = 100
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_BASE_DIR, '..', 'data', 'processed')

# 复现控制（单一种子）
seed = 3407
deterministic = True
cudnn_benchmark = False

# 训练稳定性
grad_clip_max_norm = 1.0

# 学习率衰减参数
scheduler_factor = 0.8
scheduler_patience = 5
scheduler_min_lr = 1e-5
scheduler_threshold = 5e-4
scheduler_cooldown = 1

batch_size = 256
sequence_len = 30
train_batteries = ['B0005', 'B0006']
val_batteries = ['B0007']
test_batteries = ['B0018']

input_size = 4
hidden_size = 128
num_layers = 2
dropout = 0.3

pinn_gamma = 0.9
ocv_min = 2.5
ocv_max = 4.3
ocv_hidden_size = 32
