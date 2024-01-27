import pandas as pd
import numpy as np
import os
import argparse

from anospp_analysis.util import *

def read_in_results(nn_path, vae_path):

    logging.info("Reading in result files")
    nn = pd.read_csv(nn_path, sep='\t')
    vae = pd.read_csv(vae_path, sep='\t')

    return nn, vae

def merge_results(nn, vae):

    logging.info("Merging result tables")
    results_df = pd.merge(nn, vae, how='left', on='sample_id')

    return results_df
 
def generate_consensus_mosquito(results_df):
    logging.info('finalising mosquito species assignments')

    #Get final species call
    #Samples assigned by VAE
    results_df['species_call'] = results_df.VAE_species
    results_df.loc[~results_df.species_call.isnull(), 'call_method'] = 'VAE'
    
    #Samples assigned by NN
    for level in ['fine', 'int', 'coarse']:
        leveldict = dict(zip(results_df.sample_id, results_df[f'res_{level}']))
        results_df.loc[(results_df.species_call.isnull()) & \
            (~results_df[f'res_{level}'].isnull()), 'call_method'] = f'NN_{level}'
        results_df.loc[results_df.call_method == f'NN_{level}', \
            'species_call'] = results_df.loc[results_df.call_method == \
            f'NN_{level}', 'sample_id'].map(leveldict)
        
    #Rainbow samples
    results_df.loc[(results_df.species_call.isnull()) & \
        (results_df.NN_assignment=='yes'), 'call_method'] = 'NN'
    results_df.loc[results_df.call_method=='NN', 'species_call'] = 'RAINBOW_SAMPLE'

    #Samples with too few targets
    results_df.loc[results_df.NN_assignment=='no', 'call_method'] = 'TOO_FEW_TARGETS'
    results_df.loc[results_df.NN_assignment=='no', 'species_call'] = 'TOO_FEW_TARGETS'

    assert not results_df.species_call.isnull().any(), 'some samples not assigned'
    assert not results_df.call_method.isnull().any(), 'some samples not assigned'

    return results_df


def agg(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP results merging started')

    nn, vae = read_in_results(args.nn, args.vae)
    results_df = merge_results(nn, vae)
    expanded_results_df = generate_consensus_mosquito(results_df)
    expanded_results_df.to_csv(f'{args.outdir}/anospp_results.tsv', sep='\t', index=False)


def main():
    
    parser = argparse.ArgumentParser("Merging ANOSPP results")
    parser.add_argument('--qc')
    parser.add_argument('--prep')
    parser.add_argument('--nn', help='path to nn results. Default: nn/nn_assignment.tsv', 
                        default='nn/nn_assignment.tsv')
    parser.add_argument('--vae', help='path to vae results. Default: vae/vae_assignment.tsv',
                        default='vae/vae_assignment.tsv')
    parser.add_argument('--plasm')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: results', 
                        default='results')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    agg(args)

if __name__ == '__main__':
    main()