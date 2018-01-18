import json
from collections import OrderedDict


def load_params(param_file=None, verbose=True):
    # Parameters from params.json and exp.json loaded here to override parameters set below
    if param_file is not None:
        hyper_p = json.loads(open(param_file).read(),
                             object_pairs_hook=OrderedDict)
        if verbose:
            print('Using hyper-parameters:')
            for key, value in hyper_p.items():
                print('{:25s} - {:12}'.format(key, str(value)))
            print('rest of parameters are set as default')
    parameters = {

        # for starting model from a checkpoint
        'reload_model': False,
        'prev_epochs': 0,

        # general parameters
        'batch_size': 100,
        'epochs': 1,
        'val_split': 0.1, #validation split
        'loss': 'categorical_crossentropy', # set reconstruction loss

        # convolution parameters
        'batchnorm_conv': True,
        'conv_activation': 'tanh',
        'conv_depth': 4,
        'conv_dim_depth': 8,
        'conv_dim_width': 8,
        'conv_d_growth_factor': 1.15875438383,
        'conv_w_growth_factor': 1.1758149644,

        # decoder parameters
        'gru_depth': 4,
        'rnn_activation': 'tanh',
        'recurrent_dim': 50,
        'do_tgru': True,                # use custom terminal gru layer 
        'terminal_GRU_implementation': 0, # use CPU intensive implementation; other implementation modes (1 - GPU, 2- memory) are not yet implemented
        'tgru_dropout': 0.0,
        'temperature': 1.00,            # amount of noise for sampling the final output 

        # middle layer parameters 
        'hg_growth_factor': 1.4928245388, # growth factor applied to determine size of next middle layer.
        'hidden_dim': 100,
        'middle_layer': 1,
        'dropout_rate_mid': 0.0,
        'batchnorm_mid': True,          # apply batch normalization to middle layers
        'activation': 'tanh',

        # Optimization parameters
        'lr': 0.000312087049936,
        'momentum': 0.936948773087,
        'optim': 'adam',                # optimizer to be used

        # vae parameters
        'vae_annealer_start': 22,       # Center for variational weigh annealer 
        'batchnorm_vae': False,         # apply batch normalization to output of the variational layer
        'vae_activation': 'tanh',
        'xent_loss_weight': 1.0,        # loss weight to assign to reconstruction error.
        'kl_loss_weight': 1.0,          # loss weight to assing to KL loss
        "anneal_sigmod_slope": 1.0,     # slope of sigmoid variational weight annealer
        "freeze_logvar_layer": False,   # Choice of freezing the variational layer until close to the anneal starting epoch
        "freeze_offset": 1,             # the number of epochs before vae_annealer_start where the variational layer should be unfrozen

        # property prediction parameters:
        'do_prop_pred': False,          # whether to do property prediction
        'prop_pred_depth': 3,
        'prop_hidden_dim': 36,
        'prop_growth_factor': 0.8,      # ratio between consecutive layer in property prediction
        'prop_pred_activation': 'tanh', 
        'reg_prop_pred_loss': 'mse',    # loss function to use with property prediction error for regression tasks
        'logit_prop_pred_loss': 'binary_crossentropy',  # loss function to use with property prediction for logistic tasks 
        'prop_pred_loss_weight': 0.5,
        'prop_pred_dropout': 0.0,
        'prop_batchnorm': True,

        # print output parameters
        "verbose_print": 0,

    }
    # overwrite parameters
    parameters.update(hyper_p)
    return parameters
