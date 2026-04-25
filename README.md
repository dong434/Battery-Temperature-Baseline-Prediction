# Battery Temperature Baseline Prediction

## 目录职责
- `src/train.py`：训练与验证，保存本次最佳模型和本次最后断点。
- `src/evaluate.py`：测试评估（默认读取最新最佳模型），输出测试指标和测试图。
- `src/predict.py`：部署推理占位入口（后续与 `deploy/` 联动）。
- `checkpoints/`：训练产出的模型与断点文件。
- `results/train_val_plots/`：训练/验证曲线图。
- `results/test_plots/`：测试评估图。
- `deploy/`：部署模型目录（当前仅占位）。
- `data/raw/`：原始数据。
- `data/processed/`：预处理后的张量数据。

## 推荐执行顺序
```powershell
python src/data_pipeline.py
python src/train.py
python src/evaluate.py
```

## 当前数据集划分
- 训练集：`B0005`、`B0006`
- 验证集：`B0007`
- 测试集：`B0018`

## 当前建模说明
- 主输入特征：`voltage_measured`、`time`、`ambient_temp`、`SOH`
- 训练目标：使用长度为 `30` 的时间窗口，预测窗口最后一个时刻的温度
- PINN 辅助量：预处理阶段会额外保存窗口末步的 `current / voltage / ambient / time / SOC / SOH`
