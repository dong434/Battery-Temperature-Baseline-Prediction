import argparse
import os
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from model import BatteryTemperatureLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = config.data_path
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
TEST_PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'test_plots')
BEIJING_TZ = ZoneInfo('Asia/Shanghai')


def _set_seed(seed=config.seed):
    # 固定随机性，保证评估可复现
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(config.deterministic)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def _find_latest_best_model():
    if not os.path.exists(CHECKPOINTS_DIR):
        return None
    candidates = [f for f in os.listdir(CHECKPOINTS_DIR) if f.startswith('best_model_') and f.endswith('.pth')]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(CHECKPOINTS_DIR, candidates[-1])


def _compute_metrics(preds_c, targets_c, eps=1e-6):
    errors = preds_c - targets_c
    abs_errors = np.abs(errors)

    mae_c = float(np.mean(abs_errors))
    max_ae_c = float(np.max(abs_errors))
    mre = float(np.mean(abs_errors / np.maximum(np.abs(targets_c), eps)))

    ss_res = float(np.sum((targets_c - preds_c) ** 2))
    mean_target = float(np.mean(targets_c))
    ss_tot = float(np.sum((targets_c - mean_target) ** 2))
    r2 = float('nan') if ss_tot <= eps else float(1.0 - ss_res / ss_tot)

    return mae_c, max_ae_c, mre, r2


def _save_metrics_table(fig_path, mae_c, max_ae_c, mre, r2):
    # 按表格风格输出评估结果，不画曲线
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=220)
    ax.axis('off')

    headers = ['Case #', 'MAE (°C)', 'MAX AE (°C)', 'MRE (%)', 'R2']
    row = ['1', f'{mae_c:.4f}', f'{max_ae_c:.4f}', f'{mre * 100:.4f}', f'{r2:.4f}']

    table = ax.table(
        cellText=[row],
        colLabels=headers,
        cellLoc='center',
        colLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.15, 1.9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor('#D9D9D9')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F2F2F2')

    ax.set_title('Test Metrics Table', fontsize=14, weight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def evaluate_model(model_path=None, batch_size=config.batch_size):
    _set_seed(config.seed)
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        model_path = _find_latest_best_model()
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError('未找到可用模型，请先训练或通过 --model_path 指定模型路径')

    x_test_path = os.path.join(DATA_DIR, 'test_x.pt')
    y_test_path = os.path.join(DATA_DIR, 'test_y.pt')
    scaler_path = os.path.join(DATA_DIR, 'train_scaler.gz')

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f'测试张量数据未找到: {x_test_path}, {y_test_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'训练集 scaler 未找到: {scaler_path}')

    x_test = torch.load(x_test_path, weights_only=True)
    y_test = torch.load(y_test_path, weights_only=True)
    y_scaler = joblib.load(scaler_path)['y_scaler']

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BatteryTemperatureLSTM(input_size=x_test.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    preds_scaled = []
    targets_scaled = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).view(-1, 1)
            output = model(batch_x)
            preds_scaled.append(output.cpu().numpy())
            targets_scaled.append(batch_y.cpu().numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    targets_scaled = np.concatenate(targets_scaled, axis=0)

    preds_c = y_scaler.inverse_transform(preds_scaled)
    targets_c = y_scaler.inverse_transform(targets_scaled)

    mae_c, max_ae_c, mre, r2 = _compute_metrics(preds_c, targets_c)

    run_id = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(TEST_PLOTS_DIR, f'test_eval_{run_id}.png')
    _save_metrics_table(fig_path, mae_c, max_ae_c, mre, r2)

    print(f'模型路径: {model_path}')
    print(f'平均绝对预测误差 MAE(℃): {mae_c:.6f}')
    print(f'最大绝对预测误差 MAX AE(℃): {max_ae_c:.6f}')
    print(f'平均相对预测误差 MRE(%): {mre * 100:.4f}%')
    print(f'R2: {r2:.6f}')
    print(f'测试表格图已保存: {fig_path}')

    return {
        'model_path': model_path,
        'mae_c': mae_c,
        'max_ae_c': max_ae_c,
        'mre': mre,
        'r2': r2,
        'figure_path': fig_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on test set (B0018 tail cycles).')
    parser.add_argument('--model_path', type=str, default=None, help='可选，手动指定模型路径')
    args = parser.parse_args()
    evaluate_model(model_path=args.model_path)
