## 快速开始

### 前置要求

```bash
python >= 3.8
pandas >= 1.0
scikit-learn >= 0.24
xgboost >= 1.5
numpy >= 1.19
```

### 安装依赖

```bash
pip install pandas scikit-learn xgboost numpy
```
或者使用uv

```bash
uv sync
```

### 执行训练脚本

超参数直接在训练脚本里修改。

```bash
python train_mlp.py
python train_logistic.py
python train_xgb.py
```

以XGBoost模型作为最终提交。测试结果保存为testlabel.txt,与testlabel_xgb.txt一致。