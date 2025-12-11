import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from preprocess import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')


def train_xgboost(X_train, y_train, X_val, y_val, X_test, params):

    print("=" * 60)
    print("Training XGBoost Model...")
    print("=" * 60)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    print("\nXGBoost Configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=20,
    )
    
    best_iteration = model.best_iteration
    print(f"\nBest iteration: {best_iteration}")
    
    y_val_pred_proba = model.predict(dval, iteration_range=(0, best_iteration + 1))
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print("\n" + "=" * 60)
    print("Validation Set Performance:")
    print("=" * 60)
    print(f"Accuracy:  {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1 Score:  {val_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features:")
    print("=" * 60)
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False).head(10)
    print(importance_df.to_string(index=False))
    
    y_test_pred_proba = model.predict(dtest, iteration_range=(0, best_iteration + 1))
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    return y_test_pred

def main():
    print("Loading preprocessed data...")
    preprocessor = DataPreprocessor(data_dir='.')
    X_train, y_train, X_test = preprocessor.preprocess(feature_engineering=False, remove_low_predictive=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42
    }

    y_pred = train_xgboost(X_train, y_train, X_val, y_val, X_test, params)
    
    with open('testlabel_xgb.txt', 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")
    print("Predictions saved as 'testlabel_xgb.txt'")


if __name__ == '__main__':
    main()