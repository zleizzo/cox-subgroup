import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.subgroup import bounding_box
from data.gen import synth_nonlinear
from config.constants import SKSURV_DATASETS, SKSURV_COLS, METABRIC_COLS, NASA_TURBOFAN_COLS


def scale_and_get_bounding_box(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    X = df[feature_cols]
    B = bounding_box(X)
    return B, scaler


def sorted_train_test_split(df, feature_cols, test_size=0.2, random_state=None):
    # Step 1: Random train/test split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Step 2: Sort each split by 'time'
    df_train = df_train.sort_values(by='time').reset_index(drop=True)
    df_test = df_test.sort_values(by='time').reset_index(drop=True)

    X = df_train[feature_cols].values
    Y = df_train[['failure', 'time']].to_records(index=False)

    X_test = df_test[feature_cols].values
    Y_test = df_test[['failure', 'time']].to_records(index=False)
    return X, Y, X_test, Y_test


def process_data(df, feature_cols, seed):
    B, scaler = scale_and_get_bounding_box(df, feature_cols)
    X, Y, X_test, Y_test = sorted_train_test_split(df, feature_cols, test_size=0.2, random_state=seed)
    return X, Y, X_test, Y_test, B, scaler


def load_metabric(seed):
    df = pd.read_pickle("../data/processed/metabric_processed.pkl")
    feature_cols = METABRIC_COLS
    return process_data(df, feature_cols, seed)


def load_nasa_turbofan(seed, n=None):
    """
    Load NASA Turbofan dataset with optional subsampling.

    Parameters
    ----------
    seed : int
        Random seed for train/test split
    n : int, optional
        Number of samples to load. If None, loads all available data.

    Returns
    -------
    X, Y, X_test, Y_test, B, scaler
        Processed data following repo conventions
    """
    # Map dataset name to file name
    df = pd.read_pickle('../data/processed/nasa_turbofan_combined_processed.pkl')
    df = df[df['op_setting_2'].apply(lambda x: abs(x - .84) < 0.01) & df['op_setting_3'].apply(lambda x: abs(x - 100.) < 0.1)]
    
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Use standard processing pipeline
    return process_data(df, NASA_TURBOFAN_COLS, seed)


def load_data(dataset, adjust_cols, subgp_cols, seed, n=None):
    """
    Load data and return separate feature matrices for adjustment and subgroup definition.

    Parameters
    ----------
    dataset : str
        Name of dataset to load
    adjust_cols : list of int
        Column indices (in original feature space) to use for Cox model adjustment
    subgp_cols : list of int
        Column indices (in original feature space) to use for subgroup definition
    seed : int
        Random seed for train/test split
    n : int, optional
        Sample size for synthetic datasets

    Returns
    -------
    X_adjust : array, shape (n_train, len(adjust_cols))
        Training features for Cox model adjustment
    X_subgp : array, shape (n_train, len(subgp_cols))
        Training features for subgroup definition
    Y : structured array, shape (n_train,)
        Training outcomes
    X_adjust_test : array, shape (n_test, len(adjust_cols))
        Test features for Cox model adjustment
    X_subgp_test : array, shape (n_test, len(subgp_cols))
        Test features for subgroup definition
    Y_test : structured array, shape (n_test,)
        Test outcomes
    B_subgp : array, shape (2, len(subgp_cols))
        Bounding box for subgroup feature space
    scaler : StandardScaler or None
        Fitted scaler (for inverse transform if needed)
    """
    rng = np.random.default_rng(seed=seed+2) # Used for defining local randomness.
    # +2 to seed to avoid overlap with other random seeds used in synth data generation.

    if dataset in SKSURV_DATASETS:
        X_full, Y_full = SKSURV_DATASETS[dataset]()
        df = pd.concat([X_full, pd.DataFrame(Y_full)], axis=1)
        df.dropna(inplace=True)
        new_names = {df.columns[-2]: "failure", df.columns[-1]: "time"}
        df = df.rename(columns=new_names)
        X, Y, X_test, Y_test, B, scaler = process_data(df, SKSURV_COLS[dataset], seed)

    elif dataset == 'metabric':
        X, Y, X_test, Y_test, B, scaler = load_metabric(seed)

    elif dataset == 'nasa':
        X, Y, X_test, Y_test, B, scaler = load_nasa_turbofan(seed, n)

    elif dataset == 'nonlinear':
        assert n is not None, "n must be specified for nonlinear dataset"
        d = 2

        B = np.ones((2, d))
        B[0, :] *= -1

        R = np.ones((2, d))
        R[0] *= -(1/6) ** (1/d)
        R[1] *= (1/6) ** (1/d)

        beta_in = 10 * np.ones(d)
        beta_out = 0.5 * np.ones(d)

        censor_param = None
        censor_type = 'none'

        X, Y = synth_nonlinear(n, B, R, beta_in, beta_out, seed, censor_param, censor_type)
        X_test, Y_test = synth_nonlinear(n, B, R, beta_in, beta_out, seed+1, censor_param, censor_type)
        scaler = None

    assert np.all(Y['time'] == sorted(Y['time'])), "Y['time'] is not sorted"
    assert np.all(Y_test['time'] == sorted(Y_test['time'])), "Y_test['time'] is not sorted"

    # Extract the two feature matrices
    X_adjust = X[:, adjust_cols]
    X_subgp = X[:, subgp_cols]
    X_adjust_test = X_test[:, adjust_cols]
    X_subgp_test = X_test[:, subgp_cols]

    # Bounding box only for subgroup features
    B_subgp = B[:, subgp_cols] if B.ndim > 1 else B

    return X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, B_subgp, scaler