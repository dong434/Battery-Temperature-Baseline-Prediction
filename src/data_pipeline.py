import joblib
import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def load_data(file_path, target_battery='B0005'):
    mat = loadmat(file_path)
    cycles = mat[target_battery][0, 0]['cycle'][0]

    data_list = []
    V_MAX = 4.2
    C_INITIAL = 2.0
    T_ENV = 24.0

    for i, cycle in enumerate(cycles):
        cycle_type = cycle['type'][0]
        if cycle_type == 'discharge':
            cycle_data = cycle['data'][0, 0]

            voltage_measured = cycle_data['Voltage_measured'][0]
            current_measured = cycle_data['Current_measured'][0]
            temperature_measured = cycle_data['Temperature_measured'][0]
            time = cycle_data['Time'][0]
            capacity = float(cycle_data['Capacity'][0, 0])

            soh = capacity / C_INITIAL

            for j in range(len(voltage_measured)):
                p_loss = abs(current_measured[j]) * (V_MAX - voltage_measured[j])

                data_list.append({
                    'cycle': i + 1,
                    'voltage_measured': voltage_measured[j],
                    'P_loss': p_loss,
                    'time': time[j],
                    'ambient_temp': T_ENV,
                    'SOH': soh,
                    'temperature_measured': temperature_measured[j],
                })

    return pd.DataFrame(data_list)


def _build_scaler():
    return MinMaxScaler()


def clean_data(df, x_scaler=None, y_scaler=None, fit=True):
    cycles = df['cycle']
    features = df[['voltage_measured', 'P_loss', 'time', 'ambient_temp', 'SOH']]
    labels = df['temperature_measured']

    if fit:
        if x_scaler is None:
            x_scaler = _build_scaler()
        if y_scaler is None:
            y_scaler = _build_scaler()

        scaled_x = x_scaler.fit_transform(features)
        scaled_y = y_scaler.fit_transform(labels.values.reshape(-1, 1))
    else:
        if x_scaler is None or y_scaler is None:
            raise ValueError('fit=False 时必须传入已拟合的 x_scaler 和 y_scaler')

        scaled_x = x_scaler.transform(features)
        scaled_y = y_scaler.transform(labels.values.reshape(-1, 1))

    y_values = scaled_y.flatten().astype(np.float32)
    return scaled_x, y_values, x_scaler, y_scaler, cycles


def create_sequence(features, labels, cycles, sequence_len=30):
    xs = []
    ys = []

    unique_cycle_label = np.unique(cycles)
    for cycle_id in unique_cycle_label:
        idx = np.where(cycles == cycle_id)[0]

        current_cycle_x = features[idx]
        current_cycle_y = labels[idx]

        for i in range(len(current_cycle_x) - sequence_len):
            xs.append(current_cycle_x[i:i + sequence_len])
            ys.append(current_cycle_y[i + sequence_len])

    x_tensor = torch.tensor(np.array(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(ys), dtype=torch.float32)
    return x_tensor, y_tensor


def split_tail_cycles(df, val_ratio_min=0.15, val_ratio_max=0.20):
    unique_cycles = np.sort(df['cycle'].unique())
    cycle_count = len(unique_cycles)
    if cycle_count < 2:
        raise ValueError('每个电池至少需要 2 个放电周期，才能切分训练集和验证集')

    min_val_count = int(np.ceil(cycle_count * val_ratio_min))    #向上取整，验证集数据最少取cycle_count * val_ratio_min个cycle
    max_val_count = int(np.floor(cycle_count * val_ratio_max))   #向下取整，验证集数据多取cycle_count * val_ratio_max个cycle
    val_count = max(min_val_count, max_val_count)
    val_count = max(1, min(val_count, cycle_count - 1))

    val_cycles = unique_cycles[-val_count:]
    is_val = df['cycle'].isin(val_cycles)
    train_df = df.loc[~is_val].copy()
    val_df = df.loc[is_val].copy()
    return train_df, val_df, cycle_count, val_count


def add_cycle_offset(df, cycle_offset):
    shifted_df = df.copy()
    shifted_df['cycle'] = shifted_df['cycle'].astype(int) + int(cycle_offset)
    next_offset = int(shifted_df['cycle'].max())
    return shifted_df, next_offset


def build_train_val_from_batteries(
    battery_ids,
    val_ratio_min=0.15,
    val_ratio_max=0.20,
    sequence_len=30,
):
    train_parts = []
    val_parts = []
    cycle_offset = 0

    for battery_id in battery_ids:
        file_path = os.path.join(PROJECT_ROOT, 'data', 'raw', f'{battery_id}.mat')
        battery_df = load_data(file_path, target_battery=battery_id)
        battery_train_df, battery_val_df, cycle_count, val_count = split_tail_cycles(
            battery_df,
            val_ratio_min=val_ratio_min,
            val_ratio_max=val_ratio_max,
        )

        battery_train_df, cycle_offset = add_cycle_offset(battery_train_df, cycle_offset)
        battery_val_df, cycle_offset = add_cycle_offset(battery_val_df, cycle_offset)

        train_parts.append(battery_train_df)
        val_parts.append(battery_val_df)
        print(
            f'{battery_id}: total_cycles={cycle_count}, '
            f'val_cycles={val_count}, train_cycles={cycle_count - val_count}'
        )

    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True)

    train_x, train_y, x_scaler, y_scaler, train_cycles = clean_data(train_df, fit=True)
    val_x, val_y, _, _, val_cycles = clean_data(
        val_df,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        fit=False,
    )

    train_x_tensor, train_y_tensor = create_sequence(
        train_x,
        train_y,
        train_cycles,
        sequence_len=sequence_len,
    )
    val_x_tensor, val_y_tensor = create_sequence(
        val_x,
        val_y,
        val_cycles,
        sequence_len=sequence_len,
    )
    return train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, x_scaler, y_scaler


def save_processed_train_data(X, y, x_scaler, y_scaler, save_name_x, save_name_y, save_name_scaler, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(X, os.path.join(save_dir, save_name_x))
    torch.save(y, os.path.join(save_dir, save_name_y))
    joblib.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, os.path.join(save_dir, save_name_scaler))
    print('训练集 数据保存完成')


def save_processed_vAt_data(X, y, save_name_x, save_name_y, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(X, os.path.join(save_dir, save_name_x))
    torch.save(y, os.path.join(save_dir, save_name_y))
    print('验证/测试集 数据保存完成')


if __name__ == '__main__':
    train_x_tensor, train_y_tensor, eval_x_tensor, eval_y_tensor, x_scaler, y_scaler = build_train_val_from_batteries(
        battery_ids=config.train_batteries,
        val_ratio_min=config.val_cycle_ratio_min,
        val_ratio_max=config.val_cycle_ratio_max,
        sequence_len=config.sequence_len,
    )

    print(f'train X shape: {train_x_tensor.shape}')
    print(f'train y shape: {train_y_tensor.shape}')

    save_processed_train_data(
        train_x_tensor, train_y_tensor, x_scaler, y_scaler,
        save_name_x='train_x.pt', save_name_y='train_y.pt', save_name_scaler='train_scaler.gz'
    )

    print(f'eval X shape: {eval_x_tensor.shape}')
    print(f'eval y shape: {eval_y_tensor.shape}')
    save_processed_vAt_data(
        eval_x_tensor, eval_y_tensor, save_name_x='evaluate_x.pt', save_name_y='evaluate_y.pt'
    )
