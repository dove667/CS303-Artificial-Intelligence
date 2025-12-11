import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')


def train_mlp(X_train, y_train, X_val, y_val, X_test, params):

    print("=" * 60)
    print("Training MLP Model...")
    print("=" * 60)
    
    print("\nMLP Configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)
    
    print(f"\nTraining completed!")
    print(f"Number of iterations: {model.n_iter_}")
    
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
    
    y_test_pred = model.predict(X_test)
    
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
    
    params = {
        'hidden_layer_sizes': (8, 8),  
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'batch_size': 32,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 10,
        'verbose': True,
        'alpha': 0.1
    }

    y_pred = train_mlp(X_train, y_train, X_val, y_val, X_test, params)

    with open('testlabel_mlp.txt', 'w') as f:
        for label in y_pred:
            f.write(f"{label}\n")
    print("Predictions saved as 'testlabel_mlp.txt'")

if __name__ == '__main__':
    main()
