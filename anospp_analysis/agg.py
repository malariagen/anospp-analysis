import pandas as pd
import numpy as np
import os
import argparse

from anospp_analysis.util import *

def agg(args):

    setup_logging(verbose=args.verbose)

    logging.info('ANOSPP results merging data import started')
    
    run_id, manifest_df = prep_samples(args.manifest)
    qc_df = pd.read_csv(args.qc, sep='\t')
    plasm_df = pd.read_csv(args.plasm, sep='\t')
    nn_df = pd.read_csv(args.nn, sep='\t')
    vae_df = pd.read_csv(args.vae, sep='\t')

    logging.info("Merging results tables")

    assert set(manifest_df.sample_id) == set(qc_df.sample_id), \
        'lanelets manifest and QC samples do not match'
    comb_df = pd.merge(manifest_df, qc_df, how='inner')

    assert set(comb_df.sample_id) == set(plasm_df.sample_id), \
        'plasm samples do not match QC & lanelets'
    comb_df = pd.merge(comb_df, plasm_df, how='inner')

    assert set(comb_df.sample_id) == set(nn_df.sample_id), \
        'NN samples do not match plasm, QC & lanelets'
    comb_df = pd.merge(comb_df, nn_df, how='inner')

    assert vae_df.index.isin(comb_df.index).all(), \
        'VAE samples do not match NN, plasm, QC & lanelets'
    comb_df = pd.merge(comb_df, vae_df, how='left')

    comb_df['nnovae_mosquito_species'] = comb_df.vae_species.fillna(comb_df.nn_species_call)
    is_nocall = comb_df['nnovae_mosquito_species'].isna()
    # assert ~is_nocall.any(), \
    #     f'could not find none of NN or VAE call for {comb_df[is_nocall].index.to_list()}'
    
    comb_df['nnovae_call_method'] = comb_df.nn_call_method
    comb_df.loc[
        comb_df.sample_id.isin(vae_df.sample_id),
        'nnovae_call_method'
    ] = 'VAE'

    comb_df.to_csv(args.out, sep='\t', index=True)


def main():
    
    parser = argparse.ArgumentParser('Merging ANOSPP run analysis results into a single file')
    parser.add_argument('-m', '--manifest',
                        help='path to GbS lanelets manifest tsv. Default: ../gbs_lanelets.tsv',
                        default='../gbs_lanelets.tsv')
    parser.add_argument('-q', '--qc',
                        help='path to qc summary tsv. Default: qc/sample_qc_stats.tsv',
                        default='qc/sample_qc_stats.tsv')
    parser.add_argument('-n', '--nn', 
                        help='path to NN assignment tsv. Default: nn/nn_assignment.tsv', 
                        default='nn/nn_assignment.tsv')
    parser.add_argument('-e', '--vae', 
                        help='path to VAE assignment tsv. Default: vae/vae_assignment.tsv',
                        default='vae/vae_assignment.tsv')
    parser.add_argument('-p', '--plasm', 
                        help='path to plasm assignment tsv. Default: plasm/plasm_assignment.tsv',
                        default='plasm/plasm_assignment.tsv')
    parser.add_argument('-o', '--out', 
                        help='Output aggregated sample metadata tsv. Default: anospp_results.tsv', 
                        default='anospp_results.tsv')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()

    agg(args)


if __name__ == '__main__':
    main()