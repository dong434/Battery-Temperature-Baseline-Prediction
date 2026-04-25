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

FEATURE_COLUMNS = ['voltage_measured', 'time', 'ambient_temp', 'SOH']


def load_data(file_path, target_battery='B0005'):
    mat = loadmat(file_path)
    cycles = mat[target_battery][0, 0]['cycle'][0]

    data_list = []
    c_initial = 2.0

    for i, cycle in enumerate(cycles):
        cycle_type = cycle['type'][0]
        if cycle_type != 'discharge':
            continue

        cycle_data = cycle['data'][0, 0]
        voltage_measured = cycle_data['Voltage_measured'][0]
        current_measured = cycle_data['Current_measured'][0]
        temperature_measured = cycle_data['Temperature_measured'][0]
        time = cycle_data['Time'][0]
        capacity = float(cycle_data['Capacity'][0, 0])
        ambient_temp_raw = cycle['ambient_temperature']
        ambient_temp = float(np.asarray(ambient_temp_raw).reshape(-1)[0]) if np.size(ambient_temp_raw) else 24.0

        soh = capacity / c_initial

        dt = np.diff(time, prepend=time[0])
        dt = np.maximum(dt, 0.0)
        soc_drop = np.cumsum(np.abs(current_measured) * dt) / 3600.0
        soc = np.clip(1.0 - soc_drop / max(capacity, 1e-8), 0.0, 1.0)

        for j in range(len(voltage_measured)):
            data_list.append(
                {
                    'cycle': i + 1,
                    'voltage_measured': float(voltage_measured[j]),
                    'current_measured': float(current_measured[j]),
                    'time': float(time[j]),
                    'ambient_temp': ambient_temp,
                    'SOH': soh,
                    'SOC': float(soc[j]),
                    'temperature_measured': float(temperature_measured[j]),
                }
            )

    return pd.DataFrame(data_list)


def _build_scaler():
    return MinMaxScaler()


def clean_data(df, x_scaler=None, y_scaler=None, fit=True):
    cycles = df['cycle'].to_numpy()
    features = df[FEATURE_COLUMNS]
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

    auxiliary = {
        'current_last': df['current_measured'].to_numpy(dtype=np.float32),
        'voltage_last': df['voltage_measured'].to_numpy(dtype=np.float32),
        'ambient_last': df['ambient_temp'].to_numpy(dtype=np.float32),
        'time_last': df['time'].to_numpy(dtype=np.float32),
        'soc_last': df['SOC'].to_numpy(dtype=np.float32),
        'soh_last': df['SOH'].to_numpy(dtype=np.float32),
    }

    y_values = scaled_y.flatten().astype(np.float32)
    return scaled_x.astype(np.float32), y_values, x_scaler, y_scaler, cycles, auxiliary


def create_sequence(features, labels, cycles, auxiliary, sequence_len=30):
    xs = []
    ys = []
    aux_values = {key: [] for key in auxiliary}

    unique_cycle_label = np.unique(cycles)
    for cycle_id in unique_cycle_label:
        idx = np.where(cycles == cycle_id)[0]
        current_cycle_x = features[idx]
        current_cycle_y = labels[idx]
        current_cycle_aux = {key: values[idx] for key, values in auxiliary.items()}

        for end_idx in range(sequence_len - 1, len(current_cycle_x)):
            start_idx = end_idx - sequence_len + 1
            xs.append(current_cycle_x[start_idx:end_idx + 1])
            ys.append(current_cycle_y[end_idx])
            for key in aux_values:
                aux_values[key].append(current_cycle_aux[key][end_idx])

    x_tensor = torch.tensor(np.array(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(ys), dtype=torch.float32)
    aux_tensors = {
        key: torch.tensor(np.array(values), dtype=torch.float32)
        for key, values in aux_values.items()
    }
    return x_tensor, y_tensor, aux_tensors


def add_cycle_offset(df, cycle_offset):
    shifted_df = df.copy()
    shifted_df['cycle'] = shifted_df['cycle'].astype(int) + int(cycle_offset)
    next_offset = int(shifted_df['cycle'].max())
    return shifted_df, next_offset


def build_split_from_batteries(
    battery_ids,
    sequence_len=30,
    split_name='train',
    x_scaler=None,
    y_scaler=None,
):
    split_parts = []
    cycle_offset = 0

    for battery_id in battery_ids:
        file_path = os.path.join(PROJECT_ROOT, 'data', 'raw', f'{battery_id}.mat')
        battery_df = load_data(file_path, target_battery=battery_id)
        battery_df, cycle_offset = add_cycle_offset(battery_df, cycle_offset)
        split_parts.append(battery_df)
        print(f'{split_name} {battery_id}: total_cycles={battery_df["cycle"].nunique()}')

    split_df = pd.concat(split_parts, axis=0, ignore_index=True)
    split_x, split_y, _, _, split_cycles, split_aux = clean_data(
        split_df,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        fit=False,
    )
    split_x_tensor, split_y_tensor, split_aux_tensors = create_sequence(
        split_x,
        split_y,
        split_cycles,
        split_aux,
        sequence_len=sequence_len,
    )
    return split_x_tensor, split_y_tensor, split_aux_tensors


def build_datasets_from_config(sequence_len=30):
    train_parts = []
    cycle_offset = 0

    for battery_id in config.train_batteries:
        file_path = os.path.join(PROJECT_ROOT, 'data', 'raw', f'{battery_id}.mat')
        battery_df = load_data(file_path, target_battery=battery_id)
        battery_df, cycle_offset = add_cycle_offset(battery_df, cycle_offset)
        train_parts.append(battery_df)
        print(f'train {battery_id}: total_cycles={battery_df["cycle"].nunique()}')

    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    train_x, train_y, x_scaler, y_scaler, train_cycles, train_aux = clean_data(train_df, fit=True)
    train_x_tensor, train_y_tensor, train_aux_tensors = create_sequence(
        train_x,
        train_y,
        train_cycles,
        train_aux,
        sequence_len=sequence_len,
    )

    val_x_tensor, val_y_tensor, val_aux_tensors = build_split_from_batteries(
        battery_ids=config.val_batteries,
        sequence_len=sequence_len,
        split_name='val',
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    test_x_tensor, test_y_tensor, test_aux_tensors = build_split_from_batteries(
        battery_ids=config.test_batteries,
        sequence_len=sequence_len,
        split_name='test',
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

    return {
        'train': (train_x_tensor, train_y_tensor, train_aux_tensors),
        'evaluate': (val_x_tensor, val_y_tensor, val_aux_tensors),
        'test': (test_x_tensor, test_y_tensor, test_aux_tensors),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
    }


def save_processed_train_data(X, y, auxiliary, x_scaler, y_scaler, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(X, os.path.join(save_dir, 'train_x.pt'))
    torch.save(y, os.path.join(save_dir, 'train_y.pt'))
    for key, tensor in auxiliary.items():
        torch.save(tensor, os.path.join(save_dir, f'train_{key}.pt'))
    joblib.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, os.path.join(save_dir, 'train_scaler.gz'))
    print('训练集 数据保存完成')


def save_processed_split_data(X, y, auxiliary, split_name, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(X, os.path.join(save_dir, f'{split_name}_x.pt'))
    torch.save(y, os.path.join(save_dir, f'{split_name}_y.pt'))
    for key, tensor in auxiliary.items():
        torch.save(tensor, os.path.join(save_dir, f'{split_name}_{key}.pt'))
    print(f'{split_name}_x.pt/{split_name}_y.pt 保存完成')


if __name__ == '__main__':
    datasets = build_datasets_from_config(sequence_len=config.sequence_len)

    train_x_tensor, train_y_tensor, train_aux = datasets['train']
    eval_x_tensor, eval_y_tensor, eval_aux = datasets['evaluate']
    test_x_tensor, test_y_tensor, test_aux = datasets['test']
    x_scaler = datasets['x_scaler']
    y_scaler = datasets['y_scaler']

    print(f'train X shape: {train_x_tensor.shape}')
    print(f'train y shape: {train_y_tensor.shape}')
    save_processed_train_data(
        train_x_tensor,
        train_y_tensor,
        train_aux,
        x_scaler,
        y_scaler,
    )

    print(f'eval X shape: {eval_x_tensor.shape}')
    print(f'eval y shape: {eval_y_tensor.shape}')
    save_processed_split_data(
        eval_x_tensor,
        eval_y_tensor,
        eval_aux,
        split_name='evaluate',
    )

    print(f'test X shape: {test_x_tensor.shape}')
    print(f'test y shape: {test_y_tensor.shape}')
    save_processed_split_data(
        test_x_tensor,
        test_y_tensor,
        test_aux,
        split_name='test',
    )
