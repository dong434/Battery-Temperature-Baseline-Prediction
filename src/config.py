import os

device = 'cpu'
lr = 0.001
weight_decay = 5e-5
epochs = 100
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_BASE_DIR, '..', 'data', 'processed')

# 复现控制
seed = 42
seeds = [42]
deterministic = True
cudnn_benchmark = False

# 学习率衰减参数
scheduler_factor = 0.8
scheduler_patience = 5
scheduler_min_lr = 1e-5
scheduler_threshold = 5e-4
scheduler_cooldown = 1

batch_size = 256
sequence_len = 30
train_batteries = ['B0005', 'B0006', 'B0007']
val_cycle_ratio_min = 0.15
val_cycle_ratio_max = 0.20
test_battery = 'B0018'
test_cycle_ratio = 0.15

input_size = 5
hidden_size = 128
num_layers = 2
dropout = 0.3

