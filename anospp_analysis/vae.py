import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import keras

from .util import *
from .nn import parse_seqids_series, construct_unique_kmer_table

#Variables
K = 8
LATENTDIM = 3
SEED = 374173
WIDTH = 128
DEPTH = 6

def prep_reference_index(reference_version, path_to_refversion):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_version}')

    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not \
        exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/selection_criteria.txt'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        selection_criteria.txt file'
    level, sgp, n_targets_str = open(f'{reference_path}/selection_criteria.txt').read().split('\t')
    n_targets = int(n_targets_str)

    assert os.path.isfile(f'{reference_path}/_weights.hdf5'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        _weights.hdf5 file'
    vae_weights_file = f'{reference_path}/_weights.hdf5'

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning(f'No version.txt file present for reference version {reference_version} \
                        at {reference_path}')
        version_name = 'unknown'
        
    return level, sgp, n_targets, vae_weights_file, version_name

def select_samples(comb_stats_df, hap_df, level, sgp, n_targets):
    '''
    Select the samples meeting the criteria for VAE assignment
    Based on NN assignment and number of targets
    '''
    #identify samples meeting selection criteria
    vae_samples = comb_stats_df.loc[(comb_stats_df[f'res_{level}'] == sgp) & \
                        (comb_stats_df['mosq_targets_recovered'] >= n_targets), 'sample_id']
    #subset haplotype df
    vae_hap_df = hap_df.query('sample_id in @vae_samples')
    
    logging.info(f'selected {len(vae_samples)} samples to be run through VAE')

    return vae_samples, vae_hap_df

def prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids):
    '''
    Prepare k-mer table for a single sample
    '''
    #set up empty arrays
    total_targets = kmers_unique_seqs.shape[0]
    table = np.zeros((total_targets, kmers_unique_seqs.shape[2]), dtype='int')
    n_haps = np.zeros((total_targets), dtype='int')

    for _, row in parsed_seqids.iterrows():
        #only record the first two haplotypes for each target
        if n_haps[row.target] < 2:
            n_haps[row.target] += 1
            table[row.target,:] += kmers_unique_seqs[row.target, row.uidx, :]
    #double counts for homs
    for target in np.arange(total_targets):
        if n_haps[target] == 1:
            table[target,:] *= 2
    #sum over targets
    summed_table = np.sum(table, axis=0)

    return summed_table

def prep_kmers(vae_hap_df, vae_samples, k):
    '''
    Prepare k-mer table for the samples to be run through VAE
    '''
    #translate unique sequences to k-mers
    kmers_unique_seqs = construct_unique_kmer_table(vae_hap_df, k)
    
    logging.info('generating k-mer tables for selected samples')

    #set up k-mer table
    kmers_samples = np.zeros((len(vae_samples), 4**k))

    #fill in table for samples
    for n, sample in enumerate(vae_samples):
        parsed_seqids = parse_seqids_series(vae_hap_df.loc[vae_hap_df.sample_id == sample, 
                                                           'seqid'])
        kmers_samples[n,:] = prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids)

    return(kmers_samples)

def latent_space_sampling(args):
    
    #Add noise to encoder output
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], LATENTDIM),
                                          mean=0, stddev=1., seed=SEED)
    
    return z_mean + keras.backend.exp(z_log_var) * epsilon

def define_vae_input(k):
    input_seq = keras.Input(shape=(4**k,))

    return input_seq

def define_encoder(k):
    input_seq = define_vae_input(k)
    x = keras.layers.Dense(WIDTH, activation = 'elu')(input_seq)
    for i in range(DEPTH-1):
        x = keras.layers.Dense(WIDTH, activation = 'elu')(x)
    z_mean = keras.layers.Dense(LATENTDIM)(x)
    z_log_var = keras.layers.Dense(LATENTDIM)(x)
    z = keras.layers.Lambda(latent_space_sampling, output_shape=(LATENTDIM,), \
                            name = 'z')([z_mean, z_log_var])
    encoder = keras.models.Model(input_seq, [z_mean,z_log_var,z], name = 'encoder')

    return encoder

def define_decoder(k):
    #Check whether you need the layer part here
    decoder_input = keras.layers.Input(shape=(LATENTDIM,), name='ls_sampling')
    x = keras.layers.Dense(WIDTH, activation = "linear")(decoder_input)
    for i in range(DEPTH-1):
        x = keras.layers.Dense(WIDTH, activation = "elu")(x)
    output = keras.layers.Dense(4**k, activation = "softplus")(x)
    decoder=keras.models.Model(decoder_input, output, name = 'decoder')

    return decoder

def define_vae(k):
    input_seq = define_vae_input(k)
    encoder = define_encoder(k)
    decoder = define_decoder(k)
    output_seq = decoder(encoder(input_seq)[2])
    vae = keras.models.Model(input_seq, output_seq, name='vae')

    return vae, encoder

def predict_latent_pos(kmer_table, vae_samples, k, vae_weights_file):

    '''
    Predict latent space of test samples based on reference database
    '''

    vae, encoder = define_vae(k)

    vae.load_weights(vae_weights_file)
    predicted_latent_pos = encoder.predict(kmer_table)

    predicted_latent_pos_df = pd.DataFrame(index=vae_samples, columns=['mean1', 'mean2', 'mean3',\
                                                                       'sd1', 'sd2', 'sd3'])
    for i in range(3):
        predicted_latent_pos_df[f'mean{i+1}'] = predicted_latent_pos[0][:,i]
        predicted_latent_pos_df[f'sd{i+1}'] = predicted_latent_pos[1][:,i]

    return predicted_latent_pos_df






def vae(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP VAE data import started')

    hap_df = pd.read_csv(args.haplotypes, sep='\t')
    comb_stats_df = pd.read_csv(args.manifest, sep='\t')

    level, sgp, n_targets, vae_weights_file, version_name = prep_reference_index(\
        args.reference_version, args.path_to_refversion)
    vae_samples, vae_hap_df = select_samples(comb_stats_df, hap_df, level, \
                                                       sgp, n_targets)
    kmer_table = prep_kmers(vae_hap_df, vae_samples, K)

    latent_positions_df = predict_latent_pos(kmer_table, vae_samples, K, vae_weights_file)
    latent_positions_df.to_csv(f'{args.outdir}/latent_positions.tsv', sep='\t')




    

    logging.info('All done!')

    
def main():
    
    parser = argparse.ArgumentParser("VAE assignment for samples in gambiae complex")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file with errors removed \
                        as generated by anospp-nn', required=True)
    parser.add_argument('-m', '--manifest', help='Sample assignment tsv file as generated by\
                        anospp-nn', required=True)
    parser.add_argument('-r', '--reference_version', help='Reference index version - \
                        currently a directory name. Default: gcrefv1', default='gcrefv1')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='vae')
    parser.add_argument('--path_to_refversion', help='path to reference index version.\
         Default: ref_databases', default='ref_databases')
    parser.add_argument('--no_plotting', help='Do not generate plots. Default: False', \
                        default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    vae(args)

if __name__ == '__main__':
    main()