![chemical VAE](https://github.com/aspuru-guzik-group/chemical_vae/blob/master/aux_data/banner.png?raw=true)
=============

This repository contains the framework and code for constructing a variational autoencoder (VAE) for use with molecular SMILES, as described in [doi:10.1021/acscentsci.7b00572](http://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00572), with preprint at [https://arxiv.org/pdf/1610.02415.pdf](https://arxiv.org/pdf/1610.02415.pdf).

In short, molecular SMILES are encoded into a code vector representation, and can be decoded from the code representation back to molecular SMILES. The autoencoder may also be jointly trained with property prediction to help shape the latent space. The new latent space can then be optimized upon to find the molecules with the most optimized properties of interest.

In our example, we perform encoding/decoding with the ZINC dataset, and shape the latent space on prediction on logP, QED, and SAS properties.

#### Upcoming updates:
- [ ] Updated Docker environment
- [ ] Improved tutorial

## Questions, problems?
Make a [github issue](https://github.com/aspuru-guzik-group/chemical_vae/issues/new) :smile:. Please be as clear and descriptive as possible.

## How to install
### Requirements: 
An [Anaconda python environment](https://www.anaconda.com/download) is recommend.
Check the environment.yml file, but primarily:
- Python >= 3.5
- Keras >= 2.0.0 && <= 2.0.7
- Tensorflow == 1.1
- RDKit
- Numpy

Jupyter notebook is required to run the ipynb examples.
Make sure that the [Keras backend](https://keras.io/backend/) is set to use Tensorflow

### via Anaconda (recommended way)
Create a conda enviroment:
```
conda env create -f environment.yml
source activate chemvae
python setup.py install
```
### via pip
Assuming you have all the requirements:

`pip install git+https://github.com/aspuru-guzik-group/chemical_vae.git`

## Example: ZINC dataset

This repository contains an example of how to run the autoencoder on the zinc dataset.

First, take a look at the zinc directory. Parameters are set in the following jsons
  - **exp.json**  - Sets parameters for location of data, global experimental parameters number of epochs to run, properties to predict etc. 

For a full description of all the parameters, see hyperparameters.py ; parameters set in exp.json will overwrite parameters in hyperparameters.py, and parameters set in params.json will overwrite parameters in both exp.json and hyperparameters.py

Once you have set the parameters, run the autoencoder using the command from directory with exp.json: 

`
python -m chemvae.train_vae
`

_(Make sure you copy examples directories to not overwrite the trained weights (*.h5))_

## Components
train_vae.py : main script for training variational autoencoder
    Accepts arguments -d ...
    Example of how to run (with example directory here)

- **models.py** - Library of models, contains the encoder, decoder and property prediction models.
- **tgru_k2_gpu.py** - Custom keras layer containing custom teacher forcing/sampling 
- **sampled_rnn_tf.py** - Custom rnn function for tgru_k2_gpu.py, written in tensorflow backend.
- **hyperparameters.py** - Some default parameter settings for the autoencoder
- **mol_utils.py** - library for parsing SMILES into one-hot encoding and vice versa
- **mol_callbacks.py** - library containing callbacks used by train_vae.py
  - Includes Weight_Annealer callback, which is used to update the weight of the KL loss component
- **vae_utils.py** - utility functions for an autoencoder object, used post processing.

## Authors:
This software is written by Jennifer Wei, Benjamin Sanchez-Lengeling, Dennis Sheberla, Rafael Gomez-Bomberelli, and Alan Aspuru-Guzik (alan@aspuru.com). 
It is based on the work published in https://arxiv.org/pdf/1610.02415.pdf by
 
 * [Rafa Gómez-Bombarelli](http://aspuru.chem.harvard.edu/rafa-gomez-bombarelli/),
 * [Jennifer Wei](http://aspuru.chem.harvard.edu/jennifer-wei),
 * [David Duvenaud](https://www.cs.toronto.edu/~duvenaud/),
 * [José Miguel Hernández-Lobato](https://jmhl.org/),
 * [Benjamín Sánchez-Lengeling](),
 * [Dennis Sheberla](https://www.sheberla.com/),
 * [Jorge Aguilera-Iparraguirre](http://aspuru.chem.harvard.edu/jorge-aguilera/),
 * [Timothy Hirzel](https://www.linkedin.com/in/t1m0thy),
 * [Ryan P. Adams](http://people.seas.harvard.edu/~rpa/'),
 * [Alán Aspuru-Guzik](http://aspuru.chem.harvard.edu/about-alan/)


Feel free to reach out to us with any questions! 

## Funding acknowledgements

"This work was supported by the Computational Chemical Sciences Program funded by the U.S.Department of Energy, Office of Science, Basic Energy Sciences, under Award #DE- FG02-17ER16362"
