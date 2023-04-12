import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse

from .util import *

def prep_reference_index(reference_dn):

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

    

def nn(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP NN data import started')

    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)

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