import numpy as np


def get_first_column(array_like):
    """
    Extracts the first column of an array-like object and returns it as a list.

    Parameters:
        array_like: array-like
            Input data, such as a nested list, NumPy array, or pandas DataFrame.

    Returns:
        list: The first column as a Python list.

    Raises:
        ValueError: If the input doesn't have at least one column.
    """
    if isinstance(array_like, list):
        # Nested list: Extract first column
        if not array_like or not isinstance(array_like[0], (list, tuple)):
            raise ValueError("Input is not a valid 2D array-like structure.")
        out = [row[0] for row in array_like]
    elif hasattr(array_like, "iloc"):  # For pandas DataFrame
        out = array_like.iloc[:, 0].tolist()
    elif hasattr(array_like, "shape"):  # For NumPy array
        import numpy as np
        if not isinstance(array_like, np.ndarray):
            array_like = np.array(array_like)
        out = array_like[:, 0].tolist()
    else:

        raise TypeError("Unsupported array-like object. Provide a nested list, NumPy array, or pandas DataFrame.")
    assert all(isinstance(item, str) for item in out)
    return out


def nan_safe_metric_wrapper(metric_func):
    """
    A generic wrapper to handle NaN values in y_pred and y_true for any metric function. If all values are NaN, will return NaN.

    Parameters:
        metric_func (callable): The metric function to wrap (e.g., brier_score_loss).

    Returns:
        A new function that handles NaNs and can be used with make_scorer.
    """

    def wrapped_metric(y_true, y_pred, sample_weight=None, **kwargs):
        # Convert y_true and y_pred to NumPy arrays if they are lists
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        # Create a mask to identify non-NaN values in y_pred
        mask = ~np.isnan(y_pred)

        # Apply the mask to both y_true and y_pred to drop NaN values
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        if sample_weight is not None:
            sample_weight_filtered = sample_weight[mask]
        else:
            sample_weight_filtered = None

        # Check if either filtered array is empty
        if len(y_true_filtered) == 0 or len(y_pred_filtered) == 0:
            return np.nan

        # Call the original metric function with the filtered arrays
        return metric_func(y_true_filtered, y_pred_filtered, sample_weight=sample_weight_filtered, **kwargs)

    return wrapped_metric