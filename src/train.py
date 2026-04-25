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
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

import config
from model import BatteryTemperatureLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TRAIN_VAL_PLOTS_DIR = os.path.join(RESULTS_DIR, 'train_val_plots')
BEIJING_TZ = ZoneInfo('Asia/Shanghai')
TIME_FEATURE_INDEX = 1


def _set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(config.deterministic)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def _ensure_dirs():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(TRAIN_VAL_PLOTS_DIR, exist_ok=True)


def _find_latest_checkpoint_for_seed(seed):
    if not os.path.exists(CHECKPOINTS_DIR):
        return None
    prefix = f'last_checkpoint_seed{seed}_'
    candidates = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if f.startswith(prefix) and f.endswith('.pth')
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(CHECKPOINTS_DIR, candidates[-1])


def _load_split_tensors(file_dir, split_name):
    x = torch.load(os.path.join(file_dir, f'{split_name}_x.pt'), weights_only=True)
    y = torch.load(os.path.join(file_dir, f'{split_name}_y.pt'), weights_only=True)
    current = torch.load(os.path.join(file_dir, f'{split_name}_current_last.pt'), weights_only=True)
    voltage = torch.load(os.path.join(file_dir, f'{split_name}_voltage_last.pt'), weights_only=True)
    ambient = torch.load(os.path.join(file_dir, f'{split_name}_ambient_last.pt'), weights_only=True)
    time_last = torch.load(os.path.join(file_dir, f'{split_name}_time_last.pt'), weights_only=True)
    soc = torch.load(os.path.join(file_dir, f'{split_name}_soc_last.pt'), weights_only=True)
    soh = torch.load(os.path.join(file_dir, f'{split_name}_soh_last.pt'), weights_only=True)
    return x, y, current, voltage, ambient, time_last, soc, soh


def _compute_alpha(loss_r, loss_f, params, alpha_prev, eps=1e-8):
    loss_r_grads = torch.autograd.grad(loss_r, params, retain_graph=True, allow_unused=True)
    loss_f_grads = torch.autograd.grad(loss_f, params, retain_graph=True, allow_unused=True)

    loss_r_abs = [grad.detach().abs().reshape(-1) for grad in loss_r_grads if grad is not None]
    loss_f_abs = [grad.detach().abs().reshape(-1) for grad in loss_f_grads if grad is not None]
    if not loss_r_abs or not loss_f_abs:
        return alpha_prev.detach()

    max_grad_r = torch.max(torch.cat(loss_r_abs))
    mean_grad_f = torch.mean(torch.cat(loss_f_abs))
    alpha_hat = max_grad_r / mean_grad_f.clamp_min(eps)
    alpha = (1.0 - config.pinn_gamma) * alpha_prev + config.pinn_gamma * alpha_hat
    return alpha.detach()


def train_model(
    seed=config.seed,
    file_dir=config.data_path,
    batch_size=config.batch_size,
    lr=config.lr,
    epochs=config.epochs,
    weight_decay=config.weight_decay,
    scheduler_factor=config.scheduler_factor,
    scheduler_patience=config.scheduler_patience,
    scheduler_min_lr=config.scheduler_min_lr,
    scheduler_threshold=config.scheduler_threshold,
    scheduler_cooldown=config.scheduler_cooldown,
    resume_training=False,
):
    _set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ensure_dirs()

    run_id = datetime.now(BEIJING_TZ).strftime('%Y%m%d_%H%M%S')
    best_model_path = os.path.join(CHECKPOINTS_DIR, f'best_model_seed{seed}_{run_id}.pth')
    last_checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'last_checkpoint_seed{seed}_{run_id}.pth')

    print(f'开始加载 (seed={seed})')
    scaler_path = os.path.join(file_dir, 'train_scaler.gz')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'训练集 scaler 未找到: {scaler_path}')

    train_tensors = _load_split_tensors(file_dir, 'train')
    val_tensors = _load_split_tensors(file_dir, 'evaluate')
    scalers = joblib.load(scaler_path)
    y_scaler = scalers['y_scaler']
    x_scaler = scalers['x_scaler']
    y_range = float(y_scaler.data_max_[0] - y_scaler.data_min_[0])
    time_range = float(x_scaler.data_max_[TIME_FEATURE_INDEX] - x_scaler.data_min_[TIME_FEATURE_INDEX])
    if time_range <= 0:
        raise ValueError('时间特征缩放范围必须大于 0，才能计算 dT/dt')

    train_dataset = TensorDataset(*train_tensors)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
    val_dataset = TensorDataset(*val_tensors)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_tensors[0].shape[-1]
    model = BatteryTemperatureLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        cooldown=scheduler_cooldown,
        min_lr=scheduler_min_lr,
    )

    start_epoch = 0
    best_val_mae_c = float('inf')
    alpha_state = torch.tensor(1.0, dtype=torch.float32)

    if resume_training:
        latest_ckpt = _find_latest_checkpoint_for_seed(seed)
        if latest_ckpt is not None:
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            best_val_mae_c = float(checkpoint.get('best_val_mae_c', best_val_mae_c))
            alpha_state = torch.tensor(float(checkpoint.get('alpha_state', 1.0)), dtype=torch.float32)
            print(f'已恢复训练: checkpoint={os.path.basename(latest_ckpt)}, start_epoch={start_epoch}')

    train_losses_scaled = []
    train_physics_losses = []
    val_losses_scaled = []
    val_mae_c_list = []
    alpha_history = []

    pinn_params = [param for param in model.parameters() if param.requires_grad]

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        train_running_loss = 0.0
        train_running_physics_loss = 0.0
        val_running_loss = 0.0
        val_preds_scaled = []
        val_targets_scaled = []

        model.train()
        for batch in train_loader:
            batch_x, batch_y, batch_current, batch_voltage, batch_ambient, _, batch_soc, batch_soh = batch
            batch_x = batch_x.to(device).requires_grad_(True)
            batch_y = batch_y.to(device).view(-1, 1)
            batch_current = batch_current.to(device).view(-1, 1)
            batch_voltage = batch_voltage.to(device).view(-1, 1)
            batch_ambient = batch_ambient.to(device).view(-1, 1)
            batch_soc = batch_soc.to(device).view(-1, 1)
            batch_soh = batch_soh.to(device).view(-1, 1)

            with torch.backends.cudnn.flags(enabled=False):
                output_scaled, ocv_hat = model(batch_x, soc=batch_soc, soh=batch_soh, return_aux=True)
                loss_r = criterion(output_scaled, batch_y)

                temp_grad = torch.autograd.grad(
                    output_scaled.sum(),
                    batch_x,
                    create_graph=True,
                    retain_graph=True,
                )[0][:, -1, TIME_FEATURE_INDEX:TIME_FEATURE_INDEX + 1]
            dtemp_dt = temp_grad * (y_range / time_range)

            output_c = output_scaled * y_range + float(y_scaler.data_min_[0])
            residual = (
                dtemp_dt
                + model.lambda_1 * (batch_voltage - ocv_hat) * batch_current
                + model.lambda_2 * (batch_ambient - output_c)
            )
            loss_f = torch.mean(residual.pow(2))

            alpha_state = _compute_alpha(loss_r, loss_f, pinn_params, alpha_state.to(device))
            total_loss = loss_r + alpha_state * loss_f

            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
            optimizer.step()

            train_running_loss += loss_r.item()
            train_running_physics_loss += loss_f.item()

        model.eval()
        with torch.no_grad():
            for v_batch in val_loader:
                v_batch_x, v_batch_y, _, _, _, _, _, _ = v_batch
                v_batch_x = v_batch_x.to(device)
                v_batch_y = v_batch_y.to(device).view(-1, 1)

                v_output = model(v_batch_x)
                v_loss = criterion(v_output, v_batch_y)
                val_running_loss += v_loss.item()

                val_preds_scaled.append(v_output.cpu().numpy())
                val_targets_scaled.append(v_batch_y.cpu().numpy())

        train_avg_loss_scaled = train_running_loss / len(train_loader)
        train_avg_physics_loss = train_running_physics_loss / len(train_loader)
        val_avg_loss_scaled = val_running_loss / len(val_loader)

        val_preds_scaled = np.concatenate(val_preds_scaled, axis=0)
        val_targets_scaled = np.concatenate(val_targets_scaled, axis=0)
        val_preds_c = y_scaler.inverse_transform(val_preds_scaled)
        val_targets_c = y_scaler.inverse_transform(val_targets_scaled)
        val_mae_c = float(np.mean(np.abs(val_preds_c - val_targets_c)))

        train_losses_scaled.append(train_avg_loss_scaled)
        train_physics_losses.append(train_avg_physics_loss)
        val_losses_scaled.append(val_avg_loss_scaled)
        val_mae_c_list.append(val_mae_c)
        alpha_history.append(float(alpha_state.item()))

        if val_mae_c < best_val_mae_c:
            best_val_mae_c = val_mae_c
            torch.save(model.state_dict(), best_model_path)
            print(f'本次最优更新(seed={seed}): epoch={global_epoch}, best_val_mae_c={best_val_mae_c:.6f}')

        torch.save(
            {
                'epoch': global_epoch,
                'seed': seed,
                'run_id': run_id,
                'best_val_mae_c': best_val_mae_c,
                'alpha_state': float(alpha_state.item()),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            last_checkpoint_path,
        )

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mae_c)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f'学习率衰减触发(seed={seed}): {prev_lr:.6f} -> {current_lr:.6f}')

        if global_epoch % 5 == 0:
            print(
                f'Epoch:{global_epoch} '
                f'train_loss_scaled:{train_avg_loss_scaled:.6f} '
                f'train_loss_physics:{train_avg_physics_loss:.6f} '
                f'val_loss_scaled:{val_avg_loss_scaled:.6f} '
                f'val_mae_c:{val_mae_c:.4f} '
                f'best_val_mae_c:{best_val_mae_c:.4f} '
                f'alpha:{alpha_state.item():.4f} '
                f'lambda_1:{model.lambda_1.item():.4f} '
                f'lambda_2:{model.lambda_2.item():.4f} '
                f'lr:{current_lr:.6f}'
            )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses_scaled, 'r-', label='train_loss_scaled')
    plt.plot(val_losses_scaled, 'b--', label='val_loss_scaled')
    plt.title(f'Scaled Loss Curve (seed={seed})', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_physics_losses, 'm-', label='train_loss_physics')
    plt.plot(alpha_history, 'c--', label='alpha')
    plt.title(f'PINN Loss Curve (seed={seed})', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_mae_c_list, 'g-', label='val_mae_c')
    plt.title(f'Unscaled MAE Curve (seed={seed})', fontsize=12)
    plt.xlabel('local epochs', fontsize=11)
    plt.ylabel('mae (Celsius)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    curve_name = f'train_val_seed{seed}_{run_id}.png'
    curve_path = os.path.join(TRAIN_VAL_PLOTS_DIR, curve_name)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'训练完成(seed={seed})')
    print(f'本次最佳模型: {best_model_path}')
    print(f'本次最后断点: {last_checkpoint_path}')
    print(f'train/val 曲线: {curve_path}')


if __name__ == '__main__':
    train_model(seed=config.seed, resume_training=False)
