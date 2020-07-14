import numpy as np
parameters = {
    "hist": 180,

    "n_estimators": 1,
    "max_depth": None,
    "min_samples_split": np.arange(1e1,1e4,10).astype(int),
}

search = {
    # "n_estimators": ["list",4,3],
    # "max_depth": ["list",4,3],
    "min_samples_split": ["list",4,3],
}
