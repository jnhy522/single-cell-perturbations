import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

ORIGINAL_FEATURES = ['cell_type', 'sm_name', 'sm_lincs_id','SMILES','control']

class MLUtils:
    @staticmethod
    def mrrmse(y, y_pred):
        return root_mean_squared_error(y.T, y_pred.T)
    
    @staticmethod
    def featurize_cell_sm(df: pd.DataFrame, cell_type_feature_map: dict[str, np.ndarray], sm_feature_map: dict[str, np.ndarray]):
        gene_cols = [c for c in df.columns if c not in ORIGINAL_FEATURES]
        X = df[ORIGINAL_FEATURES]
        y = df[gene_cols]

        X_tr = []
        for i, row in X.iterrows():
            cell_type_i = row.cell_type
            sm_name_i = row.sm_name
            x_cell_type = cell_type_feature_map[cell_type_i].flatten()
            x_sm_name = sm_feature_map[sm_name_i].flatten()
            x_row = np.concat([x_cell_type, x_sm_name])
            X_tr.append(x_row)

        X_tr = pd.DataFrame(np.array(X_tr))
        return X_tr, y


    @staticmethod
    def train(X_tr, y_tr, model, folds, pca_y=None, scaler=None):
        mrrmse_list = []

        for i, (train_idx, test_idx) in enumerate(folds):
                        
            X_train = X_tr.iloc[train_idx]
            y_train = y_tr.iloc[train_idx]
            X_test = X_tr.iloc[test_idx]
            y_test = y_tr.iloc[test_idx]

            if scaler is not None:
                X_train = pd.DataFrame(scaler.fit_transform(X_train))
                X_test = pd.DataFrame(scaler.transform(X_test))

            if pca_y is not None:
                y_z_train = pca_y.fit_transform(y_train)
                model.fit(X_train, y_z_train)
                y_z_hat = model.predict(X_test)
                y_hat = pca_y.inverse_transform(y_z_hat)
            else:
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)

            err = MLUtils.mrrmse(y_test, y_hat)
            # print(f"fold{i}: MRRMSE = {err}")
            mrrmse_list.append(err)

        avg_mrrmse = sum(mrrmse_list)/len(mrrmse_list)
        # print(f"average mrrmse: {avg_mrrmse}")

        return {"folds": mrrmse_list, "mean": avg_mrrmse}

    @staticmethod
    def train_logo(X_tr, y_tr, groups, model, pca_y=None, scaler=None):
        mrrmse_list = []
        logo_folds = list(LeaveOneGroupOut().split(X_tr, y_tr, groups=groups))

        for i, (train_idx, test_idx) in enumerate(logo_folds):
            name = groups[test_idx][0]

            X_train = X_tr.iloc[train_idx]
            y_train = y_tr.iloc[train_idx]
            X_test = X_tr.iloc[test_idx]
            y_test = y_tr.iloc[test_idx]

            if scaler is not None:
                X_train = pd.DataFrame(scaler.fit_transform(X_train))
                X_test = pd.DataFrame(scaler.transform(X_test))

            if pca_y is not None:
                y_z_train = pca_y.fit_transform(y_train)
                model.fit(X_train, y_z_train)
                y_z_hat = model.predict(X_test)
                y_hat = pca_y.inverse_transform(y_z_hat)
            else:
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)    

            err = MLUtils.mrrmse(y_test, y_hat)
            print(f"hold out group {name}: MRRMSE = {err}")
            mrrmse_list.append(err)

        avg_mrrmse = sum(mrrmse_list)/len(mrrmse_list)
        return {"folds": mrrmse_list, "mean": avg_mrrmse}