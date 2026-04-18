import os
import pandas as pd
import numpy as np
import json
import math
from pprint import pprint


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import ilr
from sklearn.pipeline import make_pipeline

import argparse
import joblib

def ilr_transform(X):
    X = np.asarray(X)
    X = X + 1e-12  # if zeros present
    return ilr(X)

def load_data(metagenbert_path,corresp,additional_paths_list, ilr_transform=False):
    metagenbert_data = []
    for f in sorted(os.listdir(metagenbert_path)):
        metagenbert_data.append(np.load(os.path.join(metagenbert_path, f,"abundance.npy")))
    metagenbert_data = np.array(metagenbert_data)
    if ilr_transform:
        metagenbert_data = ilr_transform(metagenbert_data)
    
    additional_data = []
    for path in additional_paths_list:
        add = pd.read_csv(path, sep=",", index_col=0)
        add = add.sort_index(axis=1)
        if ilr_transform:
            add = ilr_transform(add.values)
        else:
            add = add.values
        additional_data.append(add)
    if len(additional_data)>0:
        additional_data = np.concatenate(additional_data, axis=1)
        # Merge metagenbert data and additional data
        merged_data = np.concatenate([metagenbert_data, additional_data], axis=1)
    else:
        merged_data = metagenbert_data
    # Get labels from corresp
    labels = json.load(open(corresp, "r"))
    y = np.array([labels[f.split(".")[0]] for f in sorted(os.listdir(metagenbert_path))])
    return merged_data, y

def classify(X, y, save_path,splits=10, random_state=42):
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    model = LogisticRegressionCV(
        penalty='l1', 
        solver='liblinear', 
        cv=cv,              
        scoring='roc_auc', # Use roc-auc for model selection
        random_state=random_state,
        max_iter=1000     # Increase max iterations to ensure convergence
        )
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X, y)
    best_score = max(model.scores_[1].mean(axis=0))
    best_model = np.argmax(model.scores_[1].mean(axis=0))
    print("Best ROC AUC:", best_score, "with C =", model.Cs_[best_model], " and standard deviation =", model.scores_[1].std(axis=0)[best_model],
          ", standard error =", model.scores_[1].std(axis=0)[best_model]/math.sqrt(splits))
    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification with LASSO')
    parser.add_argument('--metagenbert_path', type=str, required=True, help='Path to MetagenBERT abundance data')
    parser.add_argument('--corresp', type=str, required=True, help='Path to correspondence file (JSON)')
    parser.add_argument('--save_path', type=str, default='model.pkl', help='Path to save the trained model')
    parser.add_argument('--additional_paths', nargs='*', default=[], help='List of additional CSV files to merge')
    parser.add_argument('--ilr_transform', action='store_true', help='Apply ILR transformation to the data')
    args = parser.parse_args()
    X, y = load_data(args.metagenbert_path, args.corresp, args.additional_paths, ilr_transform=args.ilr_transform)
    classify(X, y, args.save_path)