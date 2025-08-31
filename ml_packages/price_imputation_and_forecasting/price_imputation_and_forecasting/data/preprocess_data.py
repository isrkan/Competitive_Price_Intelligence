import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def align_encoded_store_data_with_price_data(store_data_with_dummies, price_data):
    """
    Aligns one-hot encoded store data (store-level static features) with time series price data using StoreID as the key.
    Ensures that for every row of price_data, we also have the correct store dummy variables.
    Keeps dummies separate from time series values.

    Parameters:
    - store_dummies (pd.DataFrame): One-hot encoded store metadata with StoreID as index.
    - price_data (pd.DataFrame): Price data with 'StoreID' as a column.

    Returns:
    - aligned_store_data_with_dummies (pd.DataFrame): Store-level metadata aligned row-wise with price_data
    """
    # Check price_data is indexed by StoreID
    if 'StoreID' not in price_data.index.names:
        raise ValueError("`price_data` must be indexed by 'StoreID'.")
    # Check store_dummies is indexed by StoreID
    if store_data_with_dummies.index.name != 'StoreID':
        raise ValueError("`store_dummies` must be indexed by 'StoreID'.")
    # Validate inputs
    if not isinstance(store_data_with_dummies, pd.DataFrame) or not isinstance(price_data, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    try:
        # Merge store dummies into price data, repeating dummies across all time steps of a store
        merged_df = store_data_with_dummies.merge(price_data, how='right', left_index=True, right_on='StoreID')

        # Extract only the dummy columns again, but now aligned row-wise with price_data
        aligned_store_data_with_dummies = merged_df[store_data_with_dummies.columns]
    except Exception as e:
        raise RuntimeError(f"Failed to merge store and price data: {e}")

    return aligned_store_data_with_dummies


def convert_to_numpy_inputs(df_price_input, df_store_input):
    """
    Converts input DataFrame into NumPy arrays.

    Parameters:
    - df_price_input (pd.DataFrame): Input price data features.
    - df_store_input (pd.DataFrame): Input store data features.

    Returns:
    - df_price_input: NumPy arrays of price data for training.
    - df_store_input: NumPy arrays of store metadata for training.
    """
    # Validate both are DataFrames
    if not isinstance(df_price_input, pd.DataFrame) or not isinstance(df_store_input, pd.DataFrame):
        raise TypeError("Both `df_price_input` and `df_store_input` must be pandas DataFrames.")

    try:
        # Convert DataFrames to NumPy arrays
        df_price_input = df_price_input.values
        df_store_input = df_store_input.values

    except Exception as e:
        raise RuntimeError(f"Failed to convert dataframe to numpy array: {e}")

    return df_price_input, df_store_input


def make_mask_and_replace_nan_with_predefined_value(df_price_input, fill_nan_value):
    """
    Creates a binary mask of observed (non-NaN) values and replaces NaN values in the input NumPy array with a specified value.

    Parameters:
    - df_price_input (np.ndarray): NumPy array with potential NaN values.
    - fill_nan_value (float): Value to use to replace NaNs.

    Returns:
    - filled_nan_price_input_data (np.ndarray): Same as input_data.values but NaNs replaced by fill_nan_value
    - masked_price_inputs (np.ndarray): Binary mask (1 if observed, 0 if missing)
    """
    # Input validation
    if not isinstance(df_price_input, np.ndarray):
        raise TypeError("`input_data` must be a NumPy array.")
    # Check if array is numeric
    if not np.issubdtype(df_price_input.dtype, np.number):
        raise TypeError(f"`input_data` must be a numeric array, got dtype {df_price_input.dtype}.")

    try:
        # Identify NaN locations and create a boolean mask for non-NaN values
        masked_price_inputs = ~np.isnan(df_price_input)
        # Replace NaN with a specified value
        filled_nan_price_input_data = np.where(masked_price_inputs, df_price_input, fill_nan_value)
    except Exception as e:
        raise RuntimeError(f"Error replacing NaN values in NumPy array: {e}")

    return filled_nan_price_input_data, masked_price_inputs.astype(int)


def simulate_missing_gaps(filled_nan_price_input_data, masked_price_inputs, gap_prob, max_gap, rng=None):
    """
    Simulates artificial missing gaps in observed time series for supervised training of imputation.

    Parameters:
    - filled_nan_price_input_data (np.ndarray): Original values array (no NaNs, already filled)
    - masked_price_inputs (np.ndarray): Original mask (1=observed, 0=real missing)
    - gap_prob (float): Probability of creating a simulated gap.
    - max_gap (int): Maximum length of simulated gap. Actual gap length is drawn uniformly at random between 1 and max_gap
    - rng (np.random.Generator): Random generator for reproducibility. If not passed, we create one

    Returns:
    - sim_nan_filled_nan_price_input_data (np.ndarray): Input data with simulated gaps zeroed out.
    - sim_mask_price_inputs (np.ndarray): Mask with artificial gaps applied (0 in simulated gaps) - including both real missing values and simulated gaps.
    - target_mask (np.ndarray): Mask of where simulated gaps were created (1 only at simulated positions)
    """
    # If no random generator provided, create a default one
    if rng is None:
        rng = np.random.default_rng()

    # Copy the original mask so we don't overwrite it
    sim_mask_price_inputs = masked_price_inputs.copy()
    # target_mask is all zeros at the start (no artificial gaps yet). We will mark simulated gaps with 1s.
    target_mask = np.zeros_like(masked_price_inputs)

    # Get dimensions: n = number of samples, t = time steps (sequence length)
    n, t = filled_nan_price_input_data.shape
    for i in range(n):  # for each sample (row / product series)
        if rng.random() < gap_prob:  # randomly decide whether to simulate a gap for this series
            start = rng.integers(0, t-1)  # pick a random start index for the gap
            gap_len = rng.integers(1, min(max_gap, t - start))  # random gap length, but not exceeding sequence length
            end = start + gap_len
            # Only mask if originally observed (mask=1)
            valid_indices = np.where(masked_price_inputs[i, start:end] == 1)[0]
            if len(valid_indices) > 0:
                sim_mask_price_inputs[i, start:end] = 0  # Set sim_mask to 0 to hide from the model
                target_mask[i, start:end] = 1  # supervised labels live here - loss will be computed here

    # zeros out simulated gaps from the input file
    sim_nan_filled_nan_price_input_data = filled_nan_price_input_data * sim_mask_price_inputs

    return sim_nan_filled_nan_price_input_data, sim_mask_price_inputs, target_mask


def split_train_val_test_for_imputation(df_sim_input, df_input, matrix_sim_masked_inputs, df_store_input, matrix_target_mask, test_size, val_size, random_state):
    """
    Splits time series arrays into training, validation, and testing sets.

    Parameters:
    - df_sim_input (np.ndarray): Simulated midding gaps in time series values (real and artificial NaNs filled with 0).
    - df_input (np.ndarray): Ground truth time series values (NaNs already filled with placeholder, e.g. 0).
    - matrix_sim_masked_inputs (np.ndarray): Binary mask (1=observed, 0=real missing).
    - df_store_input (np.ndarray): Store dummy features aligned with time series.
    - matrix_target_mask (np.ndarray): Binary mask where artificial gaps were simulated (1=simulated missing, 0=else).
    - test_size (float): Proportion of the dataset for the test split (0 < test_size < 1).
    - val_size (float): Proportion of the training data (after test split) used for validation (0 < val_size < 1).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train/val/test splits:
      (X_train, mask_train, target_mask_train, y_train,
       X_val,   mask_val,   target_mask_val,   y_val,
       X_test,  mask_test,  target_mask_test,  y_test)
    """
    # Input validation
    try:
        # Check array types
        for arr, name in zip(
            [df_sim_input, df_input, matrix_sim_masked_inputs, df_store_input, matrix_target_mask],
            ["df_sim_input", "df_input", "matrix_sim_masked_inputs", "df_store_input", "matrix_target_mask"]):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array, got {type(arr)} instead.")

        # Check dimensions consistency
        n, T = df_sim_input.shape
        if df_input.shape != (n, T):
            raise ValueError(f"df_input shape {df_input.shape} must equal (n,T)={(n, T)}")
        if matrix_sim_masked_inputs.shape != (n, T):
            raise ValueError(f"matrix_sim_masked_inputs shape {matrix_sim_masked_inputs.shape} must equal {(n, T)}")
        if matrix_target_mask.shape != (n, T):
            raise ValueError(f"matrix_target_mask shape {matrix_target_mask.shape} must equal {(n, T)}")

        # Check test_size and val_size ranges
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
        if not (0 < val_size < 1):
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}.")

    except Exception as e:
        raise RuntimeError(f"Input validation failed: {e}")

    # Step 1: Build the model input (X).
    # Here we stack the "df_sim_input" (time series with NaNs filled) and the "matrix_sim_masked_inputs" to build tensors for the neural networks model.
    # So if values.shape = (n_samples, seq_len, 1) and mask.shape = (n_samples, seq_len, 1), then X.shape = (n_samples, seq_len, 2).
    try:
        # X_sequence: (n, T, 2) : channel 0 -> price_with_gaps; channel 1 -> observation mask
        X_sequence = np.stack([df_sim_input, matrix_sim_masked_inputs], axis=-1)

        # X_static: (n, D) as-is (no time dimension)
        X_static = df_store_input  # (n, D)
    except Exception as e:
        raise RuntimeError(f"Failed to stack inputs into X: {e}")

    # Step 2: Define the target (y).
    # For imputation, the target is always the full ground truth values (not the input with gaps, but the original filled series).
    # y and target masks to (n, T, 1)
    y = df_input[..., None]          # (n, T, 1)
    target_mask_3d = matrix_target_mask[..., None]  # (n, T, 1)

    # Step 3: First split into (train+val) and test.
    # We only split indices here (not the arrays yet), to ensure all arrays (X, matrix_sim_masked_inputs, matrix_target_mask, y) stay aligned.
    try:
        train_val_idx, test_idx = train_test_split(
            np.arange(X_sequence.shape[0]),  # indices of samples
            test_size=test_size,
            random_state=random_state
        )
    except Exception as e:
        raise RuntimeError(f"Error during train/test split: {e}")

    # Step 4: Split the train+val further into train and validation sets.
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=random_state
        )
    except Exception as e:
        raise RuntimeError(f"Error during train/validation split: {e}")

    # Step 5: Use the indices to slice ALL arrays consistently.
    # This way we guarantee that X, matrix_masked_inputs, matrix_target_mask, y stay in sync.
    return (
        X_sequence[train_idx], X_static[train_idx], target_mask_3d[train_idx], y[train_idx],  # Training
        X_sequence[val_idx], X_static[val_idx], target_mask_3d[val_idx], y[val_idx],  # Validation
        X_sequence[test_idx], X_static[test_idx], target_mask_3d[test_idx], y[test_idx],  # Testing
    )


def prepare_imputation_model_inputs_for_predictions(df_scaled_price_input, matrix_masked_price_inputs, df_store_input):
    """
    Prepare imputation model inputs (X_sequence, X_static) for predictions.

    Parameters:
    - df_scaled_price_input (np.ndarray): Scaled price input matrix, shape (n_samples, seq_len).
    - matrix_masked_price_inputs (np.ndarray): Mask matrix for missing values, shape (n_samples, seq_len).
    - df_store_input (np.ndarray): Store-level static input features, shape (n_samples, num_static_features).

    Returns:
    - X_sequence (np.ndarray): Sequence input for the model, shape (n_samples, seq_len, 2).
    - X_static (np.ndarray): Static input for the model, shape (n_samples, num_static_features).
    """
    try:
        # Stack scaled input and mask to form sequence input
        X_sequence = np.stack([df_scaled_price_input, matrix_masked_price_inputs], axis=-1)
        # Static input is directly the store features
        X_static = df_store_input

        return X_sequence, X_static
    except Exception as e:
        raise RuntimeError(f"[prepare_model_inputs] Error preparing inputs: {e}")



def split_chronologically_train_val_test_for_forecasting(data, test_size, val_size):
    """
    Chronologically split a matrix of series into train/val/test segments.

    Parameters:
    - series_matrix (np.ndarray): Shape (n_series, T). Each row = one time series (already scaled/transformed).
    - test_size (float): Fraction of total timeline reserved for test.
    - val_size (float): Fraction of remaining timeline (after test) reserved for validation.

    Returns:
    - train_data (np.ndarray): Array of shape (n_series, train_len). Earliest part of each series.
    - val_data (np.ndarray): Array of shape (n_series, val_len). Middle part of each series.
    - test_data (np.ndarray): Array of shape (n_series, test_len). Most recent part of each series.
    """
    # Input checks
    if not isinstance(data, np.ndarray):
        raise TypeError("Expected input `data` to be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Expected 2D array (n_series, T). Got shape {}".format(data.shape))

    # Unpack dimensions
    n_series, T = data.shape  # Number of series, timeline length

    # Compute sizes of train/val/test splits
    test_len = int(T * test_size)  # Length of test set (last part of timeline)
    if test_len <= 0 or test_len >= T:
        raise ValueError("Invalid test_size resulting in length {}".format(test_len))

    train_val_len = T - test_len  # Length remaining for train + validation
    val_len = int(train_val_len * val_size)  # Fraction of remaining reserved for validation
    train_len = train_val_len - val_len  # Rest for training

    # Check that splits are valid (no empty sets)
    if train_len <= 0 or val_len <= 0:
        raise ValueError("Invalid split sizes: train={}, val={}, test={}".format(train_len, val_len, test_len))

    # Perform chronological split (no shuffling, keep time order)
    train_data = data[:, :train_len]  # first part = training
    val_data = data[:, train_len:train_len + val_len]  # middle part = validation
    test_data = data[:, train_len + val_len:]  # last part = test

    return train_data, val_data, test_data


def create_sliding_windows(data, input_lookback, horizon, min_stride, max_stride, random_state):
    """
    Create sliding windows (X, y) from time series data.

    Parameters:
    - data (np.ndarray): 2D array of shape (n_series, T) where T is the timeline length.
    - input_lookback (int): Number of timesteps in the input window (history).
    - horizon (int): Forecast horizon (how many timesteps into the future to predict).
    - stride: Step size to slide the window across the timeline.
        - min_stride (int): Minimum stride length for sliding window.
        - max_stride (int): Maximum stride length for sliding window.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X (np.ndarray): Input windows, shape (n_samples, input_lookback).
    - y (np.ndarray): Targets, shape (n_samples, horizon).
    - ids (np.ndarray): Series index for each sample, shape (n_samples,) - Each element = integer ID of which series (row in `data`) the sample came from.
                        Useful when you have many series and want to link windows back to products.
    - t0s (np.ndarray): Forecast start time index (column index in original series) for each sample. This marks where the prediction begins.
                        Useful for reconstructing predictions back into the original timeline.
    """
    # Ensure input is a numpy array with 2D shape (n_series, T)
    if not isinstance(data, np.ndarray):
        raise TypeError("Expected input `data` to be a numpy array.")
    if data.ndim != 2:
        raise ValueError("Expected 2D array (n_series, T). Got shape {}".format(data.shape))

    n_series, T = data.shape  # Number of series, timeline length

    # Random generator for reproducibility
    rng = np.random.default_rng(random_state)
    # Storage lists for collecting samples
    X_list, y_list, ids, t0s = [], [], [], []

    # Iterate over each series separately
    for s in range(n_series):
        # Last possible starting index to leave room for horizon
        last_t0 = T - horizon

        # Choose a random stride for this series
        stride = rng.integers(min_stride, max_stride + 1)

        # Slide a moving window across the timeline
        # Start at input_lookback (so we have enough history), stop at last_t0 (so we have enough future for horizon) and move forward by `stride` each time
        for t0 in range(input_lookback, last_t0 + 1, stride):
            # Input = window of past `input_lookback` values ending at t0-1
            x = data[s, t0 - input_lookback:t0]
            # Target = next `horizon` values starting at t0
            y = data[s, t0:t0 + horizon]

            # Only keep if window sizes match (in case series is too short)
            if x.shape[0] == input_lookback and y.shape[0] == horizon:
                X_list.append(x)
                y_list.append(y)
                ids.append(s)  # remember which series it came from
                t0s.append(t0)  # remember forecast start position

    # Safety check: ensure we actually created windows
    if not X_list:
        raise RuntimeError("No windows created. Check lookback, horizon, and series length.")

    # Stack lists into numpy arrays
    X = np.stack(X_list, axis=0)  # shape (n_samples, input_lookback)
    y = np.stack(y_list, axis=0)  # shape (n_samples, horizon)
    ids = np.array(ids, dtype=int)  # series IDs
    t0s = np.array(t0s, dtype=int)  # forecast start indices

    return X, y, ids, t0s


def prepare_forecasting_model_inputs_for_predictions(df_price_with_imputed_data, input_lookback):
    """
    Prepare the most recent lookback window per series for forecasting.

    Parameters:
    - df_price_with_imputed_data (np.ndarray): 2D array of shape (n_series, T), containing fully imputed + scaled time series data.
    - input_lookback (int): Number of past timesteps to feed into the forecasting model.

    Returns:
    - X_sequence (np.ndarray): Last lookback timesteps for each series, shape (n_series, input_lookback, 1).
    """
    # Validate inputs
    if not isinstance(df_price_with_imputed_data, np.ndarray):
        raise TypeError("Expected df_price_with_imputed_data to be numpy array.")
    n_series, T = df_price_with_imputed_data.shape
    if T < input_lookback:
        raise ValueError(f"Series length {T} is shorter than input_lookback {input_lookback}.")

    try:
        # Take the last input_lookback timesteps for each series
        X_sequence = df_price_with_imputed_data[:, -input_lookback:]  # shape (n_series, lookback)

        return X_sequence
    except Exception as e:
        raise RuntimeError(f"[prepare_model_inputs] Error preparing inputs: {e}")