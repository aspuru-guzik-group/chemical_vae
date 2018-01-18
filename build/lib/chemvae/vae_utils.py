# from autoencoder.utils import encode_decode as lasp
from . import mol_utils as mu
from . import hyperparameters
import random
import yaml
from .models import load_encoder, load_decoder, load_property_predictor
import numpy as np
import pandas as pd
import os
from .mol_utils import fast_verify


class VAEUtils(object):
    def __init__(self,
                 exp_file='exp.json',
                 encoder_file=None,
                 decoder_file=None,
                 directory=None):
        # files
        if directory is not None:
            curdir=os.getcwd()
            os.chdir(os.path.join(curdir,directory))
            # exp_file = os.path.join(directory, exp_file)

        # print(os.getcwd())

        # load parameters
        self.params = hyperparameters.load_params(exp_file, False)
        if encoder_file is not None:
            self.params["encoder_weights_file"] = encoder_file
        if decoder_file is not None:
            self.params["decoder_weights_file"] = decoder_file
        # char stuff
        chars = yaml.safe_load(open(self.params['char_file']))
        self.chars = chars
        self.params['NCHARS'] = len(chars)
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        # encoder, decoder
        self.enc = load_encoder(self.params)
        self.dec = load_decoder(self.params)
        self.encode, self.decode = self.enc_dec_functions()
        self.data = None
        if self.params['do_prop_pred']:
            self.property_predictor = load_property_predictor(self.params)

        # Load data without normalization as dataframe
        df = pd.read_csv(self.params['data_file'])
        df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        df = df[df.iloc[:, 0].str.len() <= self.params['MAX_LEN']]
        self.smiles = df.iloc[:, 0].tolist()
        if df.shape[1] > 1:
            self.data = df.iloc[:, 1:]

        self.estimate_estandarization()
        if directory is not None:
            os.chdir(curdir)
        return

    def estimate_estandarization(self):
        print('Standarization: estimating mu and std values ...', end='')
        # sample Z space

        smiles = self.random_molecules(size=50000)
        batch = 2500
        Z = np.zeros((len(smiles), self.params['hidden_dim']))
        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self.smiles_to_hot(sub_smiles)
            Z[chunk, :] = self.encode(one_hot, False)

        self.mu = np.mean(Z, axis=0)
        self.std = np.std(Z, axis=0)
        self.Z = self.standardize_z(Z)

        print('done!')
        return

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def unstandardize_z(self, z):
        return (z * self.std) + self.mu

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def smiles_distance_z(self, smiles, z0):
        x = self.smiles_to_hot(smiles)
        z_rep = self.encode(x)
        return np.linalg.norm(z0 - z_rep, axis=1)

    def prep_mol_df(self, smiles, z):
        df = pd.DataFrame({'smiles': smiles})
        sort_df = pd.DataFrame(df[['smiles']].groupby(
            by='smiles').size().rename('count').reset_index())
        df = df.merge(sort_df, on='smiles')
        df.drop_duplicates(subset='smiles', inplace=True)
        df = df[df['smiles'].apply(fast_verify)]
        if len(df) > 0:
            df['mol'] = df['smiles'].apply(mu.smiles_to_mol)
        if len(df) > 0:
            df = df[pd.notnull(df['mol'])]
        if len(df) > 0:
            df['distance'] = self.smiles_distance_z(df['smiles'], z)
            df['frequency'] = df['count'] / float(sum(df['count']))
            df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
            df.sort_values(by='distance', inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def z_to_smiles(self,
                       z,
                       decode_attempts=250,
                       noise_norm=0.0,
                       constant_norm=False,
                       early_stop=None):
        if not (early_stop is None):
            Z = np.tile(z, (25, 1))
            Z = self.perturb_z(Z, noise_norm, constant_norm)
            X = self.decode(Z)
            smiles = self.hot_to_smiles(X, strip=True)
            df = self.prep_mol_df(smiles, z)
            if len(df) > 0:
                low_dist = df.iloc[0]['distance']
                if low_dist < early_stop:
                    return df

        Z = np.tile(z, (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(Z)
        smiles = self.hot_to_smiles(X, strip=True)
        df = self.prep_mol_df(smiles, z)
        return df

    def enc_dec_functions(self, standardized=False):
        print('Using standarized functions? {}'.format(standardized))
        if not self.params['do_tgru']:
            def decode(z, standardized=standardized):
                if standardized:
                    return self.dec.predict(self.unstandardize_z(z))
                else:
                    return self.dec.predict(z)
        else:
            def decode(z, standardize=standardized):
                fake_shape = (z.shape[0], self.params[
                    'MAX_LEN'], self.params['NCHARS'])
                fake_in = np.zeros(fake_shape)
                if standardize:
                    return self.dec.predict([self.unstandardize_z(z), fake_in])
                else:
                    return self.dec.predict([z, fake_in])

        def encode(X, standardize=standardized):
            if standardize:
                return self.standardize_z(self.enc.predict(X)[0])
            else:
                return self.enc.predict(X)[0]

        return encode, decode

    # Now reports predictions after un-normalization.
    def predict_prop_Z(self, z):
        # both regression and logistic
        if (('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0) and
                ('logit_prop_tasks' in self.params) and (len(self.params['logit_prop_tasks']) > 0)):
            reg_pred, logit_pred = self.property_predictor.predict(z)
            if 'data_normalization_out' in self.params:
                df_norm = pd.read_csv(self.params['data_normalization_out'])
                reg_pred = reg_pred * df_norm['std'].values + df_norm['mean'].values
            return reg_pred, logit_pred
        # regression only scenario
        elif ('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0):
            reg_pred = self.property_predictor.predict(z)
            if 'data_normalization_out' in self.params:
                df_norm = pd.read_csv(self.params['data_normalization_out'])
                reg_pred = reg_pred * df_norm['std'].values + df_norm['mean'].values
            return reg_pred
        # logit only scenario
        else:
            logit_pred = self.property_predictor.predict(self.encode(z))
            return logit_pred

    # wrapper functions
    def predict_property_function(self):
        # Now reports predictions after un-normalization.
        def predict_prop(X):
            # both regression and logistic
            if (('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0) and
                    ('logit_prop_tasks' in self.params) and (len(self.params['logit_prop_tasks']) > 0)):
                reg_pred, logit_pred = self.property_predictor.predict(self.encode(X))
                if 'data_normalization_out' in self.params:
                    df_norm = pd.read_csv(self.params['data_normalization_out'])
                    reg_pred = reg_pred * df_norm['std'].values + df_norm['mean'].values
                return reg_pred, logit_pred
            # regression only scenario
            elif ('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0):
                reg_pred = self.property_predictor.predict(self.encode(X))
                if 'data_normalization_out' in self.params:
                    df_norm = pd.read_csv(self.params['data_normalization_out'])
                    reg_pred = reg_pred * df_norm['std'].values + df_norm['mean'].values
                return reg_pred

            # logit only scenario
            else:
                logit_pred = self.property_predictor.predict(self.encode(X))
                return logit_pred

        return predict_prop



    def ls_sampler_w_prop(self, size=None, batch=2500, return_smiles=False):
        if self.data is None:
            print('use this sampler only for external property files')
            return

        cols = []
        if 'reg_prop_tasks' in self.params:
            cols += self.params['reg_prop_tasks']
        if 'logit_prop_tasks' in self.params:
            cols += self.params['logit_prop_tasks']
        idxs = self.random_idxs(size)
        smiles = [self.smiles[idx] for idx in idxs]
        data = [self.data.iloc[idx] for idx in idxs]
        Z = np.zeros((len(smiles), self.params['hidden_dim']))

        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self.smiles_to_hot(sub_smiles)
            Z[chunk, :] = self.encode(one_hot)

        if return_smiles:
            return Z, data, smiles

        return Z, data


    def smiles_to_hot(self, smiles, canonize_smiles=True, check_smiles=False):
        if isinstance(smiles, str):
            smiles = [smiles]

        if canonize_smiles:
            smiles = [mu.canon_smiles(s) for s in smiles]

        if check_smiles:
            smiles = mu.smiles_to_hot_filter(smiles, self.char_indices)

        p = self.params
        z = mu.smiles_to_hot(smiles,
                             p['MAX_LEN'],
                             p['PADDING'],
                             self.char_indices,
                             p['NCHARS'])
        return z

    def hot_to_smiles(self, hot_x, strip=False):
        smiles = mu.hot_to_smiles(hot_x, self.indices_char)
        if strip:
            smiles = [s.strip() for s in smiles]
        return smiles

    def random_idxs(self, size=None):
        if size is None:
            return [i for i in range(len(self.smiles))]
        else:
            return random.sample([i for i in range(len(self.smiles))], size)

    def random_molecules(self, size=None):
        if size is None:
            return self.smiles
        else:
            return random.sample(self.smiles, size)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
