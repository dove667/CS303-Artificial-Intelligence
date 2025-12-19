## 快速开始

### 前置要求

```bash
python >= 3.8
pandas >= 1.0
scikit-learn >= 0.24
xgboost >= 1.5
numpy >= 1.19
torch >= 1.7
matplotlib >= 3.3
```

### 安装依赖

```bash
pip install pandas scikit-learn xgboost numpy torch matplotlib  
```

### 执行训练脚本

超参数直接在训练脚本里修改。

```bash
python train_mlp.py
python train_logistic.py
python train_xgb.py
```

以XGBoost模型作为最终提交。最终测试结果保存testlabel_xgb.txt。