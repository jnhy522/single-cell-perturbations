import pickle
import pandas as pd
from sklearn.model_selection import KFold


# CONSTANTS
SEED = 42
FEATURES = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']

PATH = "/Users/jennyng/Desktop/Projects/kaggle/single-cell-perturbations/data/de_train.parquet"
OUT_PATH = "/Users/jennyng/Desktop/Projects/kaggle/single-cell-perturbations/processed_data/train_test_splits.pkl"


def load_data(PATH):
    file = pd.read_parquet(PATH)
    return file

def make_splits(df, n_fold=10):
    cell_type = df.cell_type.unique()

    per_cell_test = {}
    for j, c in enumerate(cell_type):
        subset_df = df[df['cell_type']==c]  # keeps original index labels even after subsetting
        idx = subset_df.index.to_numpy()   # to convert kf.split positions back to original indices
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=SEED+j)  
        # kf.split is a generator and yields train, test positions for each fold rather than loading into memory all at once
        per_cell_test[c] = []   # initiate empty list for cell
        for _, test_pos in kf.split(idx):
            per_cell_test[c].append(idx[test_pos]) # append each fold's arrays of indices

    folds = []
    for i in range(n_fold):
        test_idx = []
        for c in cell_type:
            test_idx.extend(per_cell_test[c][i])    # appends array of indices as a flattened list
        train_idx = df.index.difference(test_idx).tolist()
        folds.append((train_idx, test_idx)) # appends (train, test) as a tuple to the list
    return folds

def save_splits(folds, path):
    with open(path, "wb") as f:
        pickle.dump(folds, f)

def load_splits(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    df = load_data(PATH)
    folds = make_splits(df, n_fold=10)
    save_splits(folds, OUT_PATH)

    loaded = load_splits(OUT_PATH)
    assert len(loaded) == len(folds)
    assert all(tr == ltr and te == lte 
               for (tr, te), (ltr, lte) in zip(folds, loaded))
    print(f"Saved and reloaded {len(loaded)} folds; splits match.")
