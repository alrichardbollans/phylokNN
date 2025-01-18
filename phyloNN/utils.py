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
    # assert all(isinstance(item, str) for item in out)
    return out