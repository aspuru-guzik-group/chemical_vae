from keras.layers import Input, Lambda
from keras.layers.core import Dense, Flatten, RepeatVector, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from keras.layers.merge import Concatenate
from .tgru_k2_gpu import TerminalGRU


# =============================
# Encoder functions
# =============================
def encoder_model(params):
    # K_params is dictionary of keras variables
    x_in = Input(shape=(params['MAX_LEN'], params[
        'NCHARS']), name='input_molecule_smi')

    # Convolution layers
    x = Convolution1D(int(params['conv_dim_depth'] *
                          params['conv_d_growth_factor']),
                      int(params['conv_dim_width'] *
                          params['conv_w_growth_factor']),
                      activation='tanh',
                      name="encoder_conv0")(x_in)
    if params['batchnorm_conv']:
        x = BatchNormalization(axis=-1, name="encoder_norm0")(x)

    for j in range(1, params['conv_depth'] - 1):
        x = Convolution1D(int(params['conv_dim_depth'] *
                              params['conv_d_growth_factor'] ** (j)),
                          int(params['conv_dim_width'] *
                              params['conv_w_growth_factor'] ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))(x)
        if params['batchnorm_conv']:
            x = BatchNormalization(axis=-1,
                                   name="encoder_norm{}".format(j))(x)

    x = Flatten()(x)

    # Middle layers
    if params['middle_layer'] > 0:
        middle = Dense(int(params['hidden_dim'] *
                           params['hg_growth_factor'] ** (params['middle_layer'] - 1)),
                       activation=params['activation'], name='encoder_dense0')(x)
        if params['dropout_rate_mid'] > 0:
            middle = Dropout(params['dropout_rate_mid'])(middle)
        if params['batchnorm_mid']:
            middle = BatchNormalization(axis=-1, name='encoder_dense0_norm')(middle)

        for i in range(2, params['middle_layer'] + 1):
            middle = Dense(int(params['hidden_dim'] *
                               params['hg_growth_factor'] ** (params['middle_layer'] - i)),
                           activation=params['activation'], name='encoder_dense{}'.format(i))(middle)
            if params['dropout_rate_mid'] > 0:
                middle = Dropout(params['dropout_rate_mid'])(middle)
            if params['batchnorm_mid']:
                middle = BatchNormalization(axis=-1,
                                            name='encoder_dense{}_norm'.format(i))(middle)
    else:
        middle = x

    z_mean = Dense(params['hidden_dim'], name='z_mean_sample')(middle)

    # return both mean and last encoding layer for std dev sampling
    return Model(x_in, [z_mean, middle], name="encoder")


def load_encoder(params):
    # Need to handle K_params somehow... 
    # Also are we going to be able to save this layer?
    # encoder = encoder_model(params, K.constant(0))
    # encoder.load_weights(params['encoder_weights_file'])
    # return encoder
    # !# not sure if this is the right format
    return load_model(params['encoder_weights_file'])


# ===========================================
# Decoder functions
# ===========================================


def decoder_model(params):
    z_in = Input(shape=(params['hidden_dim'],), name='decoder_input')
    true_seq_in = Input(shape=(params['MAX_LEN'], params['NCHARS']),
                        name='decoder_true_seq_input')

    z = Dense(int(params['hidden_dim']),
              activation=params['activation'],
              name="decoder_dense0"
              )(z_in)
    if params['dropout_rate_mid'] > 0:
        z = Dropout(params['dropout_rate_mid'])(z)
    if params['batchnorm_mid']:
        z = BatchNormalization(axis=-1,
                               name="decoder_dense0_norm")(z)

    for i in range(1, params['middle_layer']):
        z = Dense(int(params['hidden_dim'] *
                      params['hg_growth_factor'] ** (i)),
                  activation=params['activation'],
                  name="decoder_dense{}".format(i))(z)
        if params['dropout_rate_mid'] > 0:
            z = Dropout(params['dropout_rate_mid'])(z)
        if params['batchnorm_mid']:
            z = BatchNormalization(axis=-1,
                                   name="decoder_dense{}_norm".format(i))(z)

    # Necessary for using GRU vectors
    z_reps = RepeatVector(params['MAX_LEN'])(z)

    # Encoder parts using GRUs
    if params['gru_depth'] > 1:
        x_dec = GRU(params['recurrent_dim'],
                    return_sequences=True, activation='tanh',
                    name="decoder_gru0"
                    )(z_reps)

        for k in range(params['gru_depth'] - 2):
            x_dec = GRU(params['recurrent_dim'],
                        return_sequences=True, activation='tanh',
                        name="decoder_gru{}".format(k + 1)
                        )(x_dec)

        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([x_dec, true_seq_in])
        else:
            x_out = GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final')(x_dec)

    else:
        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([z_reps, true_seq_in])
        else:
            x_out = GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final'
                        )(z_reps)

    if params['do_tgru']:
        return Model([z_in, true_seq_in], x_out, name="decoder")
    else:
        return Model(z_in, x_out, name="decoder")


def load_decoder(params):
    if params['do_tgru']:
        return load_model(params['decoder_weights_file'], custom_objects={'TerminalGRU': TerminalGRU})
    else:
        return load_model(params['decoder_weights_file'])


##====================
## Middle part (var)
##====================

def variational_layers(z_mean, enc, kl_loss_var, params):
    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.

    def sampling(args):
        z_mean, z_log_var = args

        epsilon = K.random_normal_variable(shape=(params['batch_size'], params['hidden_dim']),
                                           mean=0., scale=1.)
        # insert kl loss here

        z_rand = z_mean + K.exp(z_log_var / 2) * kl_loss_var * epsilon
        return K.in_train_phase(z_rand, z_mean)


    # variational encoding
    z_log_var_layer = Dense(params['hidden_dim'], name='z_log_var_sample')
    z_log_var = z_log_var_layer(enc)

    z_mean_log_var_output = Concatenate(
        name='z_mean_log_var')([z_mean, z_log_var])

    z_samp = Lambda(sampling)([z_mean, z_log_var])

    if params['batchnorm_vae']:
        z_samp = BatchNormalization(axis=-1)(z_samp)

    return z_samp, z_mean_log_var_output


# ====================
# Property Prediction
# ====================

def property_predictor_model(params):
    if ('reg_prop_tasks' not in params) and ('logit_prop_tasks' not in params):
        raise ValueError('You must specify either regression tasks and/or logistic tasks for property prediction')

    ls_in = Input(shape=(params['hidden_dim'],), name='prop_pred_input')

    prop_mid = Dense(params['prop_hidden_dim'],
                     activation=params['prop_pred_activation'])(ls_in)
    if params['prop_pred_dropout'] > 0:
        prop_mid = Dropout(params['prop_pred_dropout'])(prop_mid)

    if params['prop_pred_depth'] > 1:
        for p_i in range(1, params['prop_pred_depth']):
            prop_mid = Dense(int(params['prop_hidden_dim'] *
                                 params['prop_growth_factor'] ** p_i),
                             activation=params['prop_pred_activation'],
                             name="property_predictor_dense{}".format(p_i)
                             )(prop_mid)
            if params['prop_pred_dropout'] > 0:
                prop_mid = Dropout(params['prop_pred_dropout'])(prop_mid)
            if 'prop_batchnorm' in params and params['prop_batchnorm']:
                prop_mid = BatchNormalization()(prop_mid)

    # for regression tasks
    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        reg_prop_pred = Dense(len(params['reg_prop_tasks']), activation='linear',
                              name='reg_property_output')(prop_mid)

    # for logistic tasks 
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0):
        logit_prop_pred = Dense(len(params['logit_prop_tasks']), activation='sigmoid',
                                name='logit_property_output')(prop_mid)

    # both regression and logistic
    if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0) and
            ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0)):

        return Model(ls_in, [reg_prop_pred, logit_prop_pred], name="property_predictor")

        # regression only scenario
    elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        return Model(ls_in, reg_prop_pred, name="property_predictor")

        # logit only scenario
    else:
        return Model(ls_in, logit_prop_pred, name="property_predictor")


def load_property_predictor(params):
    return load_model(params['prop_pred_weights_file'])
