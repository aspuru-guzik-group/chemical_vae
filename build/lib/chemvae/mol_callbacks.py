from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import pandas as pd
from keras import backend as K
import os


class RmseCallback(Callback):
    def __init__(self, X_test, Y_test, params, df_norm: pd.DataFrame = None):
        super(RmseCallback, self).__init__()
        self.df_norm = df_norm
        self.X_test = X_test
        self.Y_test = Y_test
        self.config = params

    def on_epoch_end(self, epoch, logs=None):
        df_norm = self.df_norm
        X_test = self.X_test
        Y_test = self.Y_test
        y_pred = self.model.predict(X_test,self.config['batch_size'])
        if type(y_pred) is list:
            if 'reg_prop_tasks' in self.config and 'logit_prop_tasks' in self.config:
                y_pred = y_pred[-2]
            elif 'reg_prop_tasks' in self.config:
                y_pred = y_pred[-1]
        if df_norm is not None:
            y_pred = y_pred * df_norm['std'].values + df_norm['mean'].values
            Y_test = Y_test * df_norm['std'].values + df_norm['mean'].values

        rmse = np.sqrt(np.mean(np.square(y_pred - Y_test), axis=0))
        mae = np.mean(np.abs(y_pred - Y_test), axis=0)
        if df_norm is not None:
            df_norm['rmse'] = rmse
            df_norm['mae'] = mae
            print("RMSE test set:", df_norm['rmse'].to_dict())
            print("MAE test set:", df_norm['mae'].to_dict())
        else:
            if "reg_prop_tasks" in self.config:
                print("RMSE test set:", self.config["reg_prop_tasks"], rmse)
                print("MAE test set:", self.config["reg_prop_tasks"], mae)
            else:
                print("RMSE test set:", rmse)
                print("MAE test set:", mae)


class WeightAnnealer_epoch(Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, schedule, weight, weight_orig, weight_name):
        super(WeightAnnealer_epoch, self).__init__()
        self.schedule = schedule
        self.weight_var = weight
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)
        new_value = new_weight * self.weight_orig
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))
        assert type(
            new_weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.weight_var, new_value)


# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num):
    return float(1)


def sigmoid_schedule(time_step, slope=1., start=None):
    return float(1 / (1. + np.exp(slope * (start - float(time_step)))))


def sample(a, temperature=0.01):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class EncoderDecoderCheckpoint(ModelCheckpoint):
    """Adapted from ModelCheckpoint, but for saving Encoder, Decoder and property
    """

    def __init__(self, encoder_model, decoder_model, params, prop_pred_model=None,
                 prop_to_monitor='val_x_pred_categorical_accuracy', save_best_only=True, monitor_op=np.greater,
                 monitor_best_init=-np.Inf):
        # Saves models at the end of every epoch if they are better than previous models
        # prop_to_montior : a property that is a valid name in the model
        # monitor_op : The operation to use when monitoring the property 
        #    (e.g. accuracy to be maximized so use np.greater, loss to minimized, so use np.loss)
        # monitor_best_init : starting point for monitor (use -np.Inf for maximization tests, and np.Inf for minimization tests)

        self.p = params
        super(ModelCheckpoint, self).__init__()
        self.save_best_only = save_best_only
        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init
        self.verbose = 1
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.prop_pred_model = prop_pred_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # self.epochs_since_last_save += 1
        # if self.epochs_since_last_save >= self.period:
        #    self.epochs_since_last_save = 0
        # filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model'
                          % (epoch, self.monitor, self.best, current))
                self.best = current
                self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}.h5'.format(epoch)))
                self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}.h5'.format(epoch)))
                if self.prop_pred_model is not None:
                    self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_{}.h5'.format(epoch)))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' %
                          (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to ' % (epoch))
            self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}.h5'.format(epoch)))
            self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}.h5'.format(epoch)))
            if self.prop_pred_model is not None:
                self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_{}.h5'.format(epoch)))
