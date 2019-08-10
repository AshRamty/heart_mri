log_config = {
        'log_dir' : './run_logs',
        'run_name' : 'search_lr_l2'
        }

tuner_config = {
        'max_search': 1,
}
search_space = {
      'l2': {'range': [0.0000001, 0.1], 'scale':'log'},          
      'lr': {'range': [0.0000001, 0.1], 'scale': 'log'},  
      #'lr': {'range': [0.0004339], 'scale':'linear'},
      #'l2': {'range': [2.2578e-05], 'scale': 'linear'},
        }
em_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    "pretrained": False,
    # Network
    # The first value is the output dim of the input module (or the sum of
    # the output dims of all the input modules if multitask=True and
    # multiple input modules are provided). The last value is the
    # output dim of the head layer (i.e., the cardinality of the
    # classification task). The remaining values are the output dims of
    # middle layers (if any). The number of middle layers will be inferred
    # from this list.
    "relu": False,
    "skip_head": True,
    "batchnorm": False,
    "dropout": 0.0,
    # GPU
    "use_cuda": True,
    "device": "cuda",
    # MODEL CLASS
    "num_layers": 3,
    "hidden_size": 128,
    "fc_size" : 30,
    "num_classes": 2,
    # LOSS CONFIG
    "loss_weights": [0.9, 0.1],
    "clean_up": False,
    # TRAINING
    "train_config": {
        # for visual loss regularization
        # Display
        "print_every": 1,  # Print after this many epochs
        "disable_prog_bar": False,  # Disable progress bar each epoch
        # Dataloader
        "data_loader_config": {"batch_size": 10, 
                                "num_workers": 16, 
                                "sampler": None
        },
       # Loss weights
        "loss_weights": [0.9, 0.1],
        # Train Loop
        "n_epochs": 10,
        # 'grad_clip': 0.0,
         "l2": 0.0,
         #"lr": 0.01,
        "validation_metric": "accuracy",
        "validation_freq": 1,
        # Evaluate dev for during training every this many epochs
        # Optimizer
        "optimizer_config": {
            "optimizer": "adam",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
            # Optimizer - Adam
            "adam_config": {"betas": (0.9, 0.999)},
        },
        # Scheduler
        "scheduler_config": {
            "scheduler": "reduce_on_plateau",
            # ['constant', 'exponential', 'reduce_on_plateu']
            # Freeze learning rate initially this many epochs
            "lr_freeze": 0,
            # Scheduler - exponential
            "exponential_config": {"gamma": 0.9},  # decay rate
            # Scheduler - reduce_on_plateau
            "plateau_config": {
                "factor": 0.5,
                "patience": 1,
                "threshold": 0.0001,
                "min_lr": 1e-5,
            },
        },
        # Checkpointer
        "checkpoint": True,
        "checkpoint_config": {
            "checkpoint_min": -1,
            # The initial best score to beat to merit checkpointing
            "checkpoint_runway": 0,
            # Don't start taking checkpoints until after this many epochs
        },
    },
}