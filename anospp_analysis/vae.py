import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import itertools

from util import *
from nn import construct_kmer_dict, parse_seqids_series, construct_unique_kmer_table

def prep_reference_index(reference_version, path_to_refversion):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_version}')

    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not \
        exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/selection_criteria.tsv'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        selection_criteria.tsv file'
    selection_criteria = pd.read_csv(f'{reference_path}/selection_criteria.tsv', sep='\t')

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning(f'No version.txt file present for reference version {reference_version} \
                        at {reference_path}')
        version_name = 'unknown'
        
    return(selection_criteria, version_name)

def select_samples_from_file(comb_stats_df, hap_df, config_file):
    
    level, sgp, n_targets = open(config_file).read().split('\t')

    return select_samples(comb_stats_df, hap_df, level, sgp, n_targets)

def select_samples_from_df(comb_stats_df, hap_df, selection_criteria):
    
    #Unwrap selection criteria 
    level = selection_criteria.loc[0,'level']
    sgp = selection_criteria.loc[0,'sgp']
    n_targets = int(selection_criteria.loc[0,'n_targets'])

    return select_samples(comb_stats_df, hap_df, level, sgp, n_targets)

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
    
    logging.info(f'Selected {len(vae_samples)} samples to be run through VAE')

    return(vae_samples, vae_hap_df)

def prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids):
    '''
    Prepare k-mer table for a single sample
    '''
    #set up empty arrays
    table = np.zeros((kmers_unique_seqs.shape[0], kmers_unique_seqs.shape[2]), dtype='int')
    n_haps = np.zeros((kmers_unique_seqs.shape[0]), dtype='int')

    for _, row in parsed_seqids.iterrows():
        #only record the first two haplotypes for each target
        if n_haps[row.target] < 2:
            n_haps[row.target] += 1
            table[row.target,:] += kmers_unique_seqs[row.target, row.uidx, :]

    

def prep_kmers(vae_hap_df, vae_samples, k):
    '''
    Prepare k-mer table for the samples to be run through VAE
    '''
    #translate unique sequences to k-mers
    kmers_unique_seqs = construct_unique_kmer_table(vae_hap_df, k)
    
    logging.info('Generate k-mer tables for selected samples')


    #Construct per sample kmer table
    table = np.zeros((len(samples), 4**k), dtype='int')
    amplified = np.zeros((len(samples), a), dtype='int')

    for e, smp in enumerate(samples):
        smptable = np.zeros((a, 4**k))
        smpseq = seq.loc[seq.s_Sample == smp]
        hapcopies = np.zeros(a)
        for r in smpseq.index:
            combUID = smpseq.loc[r, 'combUID']
            t, u = int(str.split(combUID, '-')[0]), int(str.split(combUID, '-')[1])
            if hapcopies[t] < 2:
                smptable[t,:] += kmer8[t, u, :]
                hapcopies[t] += 1
        smpamplified = np.where(np.isin(hapcopies,[1,2]), 1, 0)
        for t in np.arange(a):
            #double homozygotes
            if hapcopies[t] == 1:
                smptable[t,:] *= 2
        table[e,:] = np.sum(smptable, axis=0)
        amplified[e,:] = smpamplified


    
    
    


def vae(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP VAE data import started')

    hap_df = pd.read_csv(args.haplotypes, sep='\t')
    comb_stats_df = pd.read_csv(args.manifest, sep='\t')

    selection_criteria, version_name = prep_reference_index(args.reference_version, \
                                                            args.path_to_refversion)
    vae_samples, vae_hap_df = select_samples(comb_stats_df, hap_df, selection_criteria)
    kmer_table = prep_kmers(vae_hap_df, vae_samples, 8)




    

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
         Default: test_data', default='test_data')
    parser.add_argument('--no_plotting', help='Do not generate plots. Default: False', \
                        default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    vae(args)

if __name__ == '__main__':
    main()