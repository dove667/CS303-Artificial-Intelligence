import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, params):

    print("=" * 60)
    print("Training Logistic Regression Model...")
    print("=" * 60)
    
    print("\nLogistic Regression Configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    print(f"\nTraining completed!")
    print(f"Number of iterations: {model.n_iter_[0]}")
    
    y_val_pred = model.predict(X_val)
    
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
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    })
    feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    y_test_pred = model.predict(X_test)
    
    return y_test_pred


def main():
    print("Loading preprocessed data...")
    preprocessor = DataPreprocessor(data_dir='.')
    X_train, y_train, X_test = preprocessor.preprocess(feature_engineering=False, remove_low_predictive=False, one_hot_encoding=True, scale=True)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    params = {
        'C': 0.1,  
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'verbose': 1,
        'n_jobs': -1,  
        'class_weight': 'balanced'
    }

    y_pred = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, params)
  
    with open('testlabel_logistic.txt', 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")
    print("Predictions saved as 'testlabel_logistic.txt'")

if __name__ == '__main__':
    main()
