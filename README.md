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
