parameters = {
    "hist": 180,

    # model hyperparameters
    "n_features_emb": 64,
    "n_hidden_rnn": 128,
    "n_layers_rnn": 1,
    "reverse_time": False,
    "bidirectional": True,

    # training_old hyperparameters
    "emb_dropout": 0.0,
    "ctx_dropout": 0.0,
    "dropout_layer": 0.0,
    "epochs": 5000,
    "batch_size": 50,
    "lr": 1e-3,
    "l2": 0.0,
    # "l2": 1e-4,
    "patience": 25,
}

search = {
}
