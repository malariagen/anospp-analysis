import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import itertools

from .util import *

def prep_mosquito_haps(hap_df):
    '''
    prepare mosquito haplotype dataframe
    remove plasmodium haplotypes
    change targets to integers
    returns haplotype dataframe
    '''

    logging.info('Prepare mosquito haplotypes')

    hap_df = hap_df.astype({'target': str})
    hap_df = hap_df[hap_df.target.isin(MOSQ_TARGETS)]
    hap_df = hap_df.astype({'target': int})
    hap_df.reset_index(inplace=True, drop=True)

    return(hap_df)

def prep_reference_index(reference_dn):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'Importing reference index {reference_dn}')

    reference_path = f'test_data/{reference_dn}/'

    assert os.path.isdir(reference_path), f'reference version {reference_dn} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}haplotypes.tsv'), f'reference version {reference_dn} at {reference_path} does not contain required haplotypes.tsv file'
    ref_hap_df = pd.read_csv(f'{reference_path}haplotypes.tsv', sep='\t')

    assert os.path.isfile(f'{reference_path}allele_freq_coarse.npy'), f'reference version {reference_dn} at {reference_path} does not contain required allele_freq_coarse.npy file'
    af_c = np.load(f'{reference_path}/allele_freq_coarse.npy')
    assert os.path.isfile(f'{reference_path}allele_freq_int.npy'), f'reference version {reference_dn} at {reference_path} does not contain required allele_freq_int.npy file'
    af_i = np.load(f'{reference_path}/allele_freq_int.npy')
    assert os.path.isfile(f'{reference_path}allele_freq_fine.npy'), f'reference version {reference_dn} at {reference_path} does not contain required allele_freq_fine.npy file'
    af_f = np.load(f'{reference_path}/allele_freq_fine.npy')

    assert os.path.isfile(f'{reference_path}sgp_coarse.txt'), f'reference version {reference_dn} at {reference_path} does not contain required sgp_coarse.txt file'
    sgp_c = []
    with open(f'{reference_path}sgp_coarse.txt', 'r') as fn:
        for line in fn:
            sgp_c.append(line.strip())

    assert os.path.isfile(f'{reference_path}sgp_int.txt'), f'reference version {reference_dn} at {reference_path} does not contain required sgp_int.txt file'
    sgp_i = []
    with open(f'{reference_path}sgp_int.txt', 'r') as fn:
        for line in fn:
            sgp_i.append(line.strip())

    assert os.path.isfile(f'{reference_path}sgp_fine.txt'), f'reference version {reference_dn} at {reference_path} does not contain required sgp_fine.txt file'
    sgp_f = []
    with open(f'{reference_path}sgp_fine.txt', 'r') as fn:
        for line in fn:
            sgp_f.append(line.strip())

    
    ref_hap_df['coarse_sgp'] = pd.Categorical(ref_hap_df['coarse_sgp'], sgp_c, ordered=True)
    ref_hap_df['intermediate_sgp'] = pd.Categorical(ref_hap_df['intermediate_sgp'], sgp_i, ordered=True)
    ref_hap_df['fine_sgp'] = pd.Categorical(ref_hap_df['fine_sgp'], sgp_f, ordered=True)


    return(ref_hap_df, af_c, af_i, af_f)

def construct_kmer_dict(k):
    '''
    construct a k-mer dict
    associating each unique k-mer of length k with a unique non-negative integer <4**k
    bases are written in capitals
    returns a dictionary
    '''
    labels = []
    for i in itertools.product('ACGT', repeat=k):
        labels.append(''.join(i))
    kmerdict = dict(zip(labels, np.arange(4**k)))
    return(kmerdict)    

def construct_unique_kmer_table(hap_df, k):
    '''
    constructs a k-mer table of dimensions n_amp * maxallele * 4**k
    represting the k-mer table of each unique sequence in the dataframe
    maxallele is the maximum number of unique sequences per target
    n_amp is the number of mosquito targets
    input: k=k (length of k-mers), hap_df=dataframe with haplotypes
    output: k-mer table representing each unique haplotype in the hap dataframe
    '''

    logging.info('Translate unique sequences to k-mers')

    maxallele = hap.groupby('target')['seqid'].nunique().max()
    kmerdict = construct_kmer_dict(k)
    
    uniqueseq = hap[['seqid', 'consensus']].drop_duplicates()

    table = np.zeros((len(MOSQ_TARGETS), maxallele, 4**k), dtype='int')
    for r in seq.index:
        seqid = str.split(uniqueseq.loc[r,'seqid'], '-')
        t, u = int(seqid[0]), int(seqid[1])
        sq = uniqueseq.loc[r,'consensus']
        for i in np.arange(len(sq)-(k-1)):
            table[t,u,kmerdict[sq[i:i+k]]] += 1
    return(table)




def nn(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP NN data import started')

    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)
    hap_df = prep_mosquito_haps(hap_df)
    test_samples = comb_stats_df.loc[comb_stats_df.mosq_targets_recovered >= 10, 'sample_id'].values

    ref_hap_df, af_c, af_i, af_f = prep_reference_index(args.reference)

    

def main():
    
    parser = argparse.ArgumentParser("NN assignment for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-r', '--reference', help='Reference index version. Default: nn1.0', default='nn1.0')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='nn')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    nn(args)

if __name__ == '__main__':
    main()