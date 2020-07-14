import numpy as np
parameters = {
    "hist": 180,

    "n_estimators":np.array([1e0,5e0,1e1,5e1,1e2,5e2,1e3,5e3]).astype(int),
    "max_depth":None,
    "min_samples_split": np.r_[2, np.arange(5,150,5)].astype(int),
}

search = {
    "n_estimators":["list",4,3],
    # "max_depth":["list",4,3],
    "min_samples_split": ["list",4,3],
}
