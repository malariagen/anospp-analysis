import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import itertools
import logging
from Bio import SeqIO
import sys
import argparse
import os

def prep_hap(hap_fn):

    hap_df = pd.read_csv(hap_fn, sep='\t')

    # compatibility with old style haplotype column names
    hap_df.rename(columns=({
        's_Sample':'sample_id',
        'frac_reads':'reads_fraction'
        }), 
    inplace=True)

    return hap_df

def prep_samples(samples_fn):

    samples_df = pd.read_csv(samples_fn, sep='\t')

    # compatibility with old style samples column names
    samples_df.rename(columns=({
        'Replicate':'sample_id',
        'Run':'run',
        'Lane':'lane',
        'Replicate':'replicate'
        }), 
    inplace=True)

    return samples_df

def prep_stats(stats_fn, hap_df, samples_df):
    '''
    Add plotting columns to original DADA2 stats table
    '''

    stats_df = pd.read_csv(stats_fn, sep='\t', index_col=0)
    # fix samples order
    print(samples_df.columns)
    stats_df = stats_df.reindex(samples_df['sample_id'])
    # denoising happens for F and R reads independently, we take minimum of those 
    # as an estimate for retained read count
    stats_df['denoised'] = stats_df[['denoisedF','denoisedR']].min(axis=1)
    # targets per sample - assume same samples as in haplotypes
    stats_df['targets_recovered'] = hap_df.groupby('sample_id').target.nunique()
    assert ~stats_df.targets_recovered.isna().any(), 'Could not calculate targets_recovered for all samples'
    # plate ids
    stats_df['plate_id'] = samples_df.set_index('sample_id').plate_id
    assert ~stats_df.plate_id.isna().any(), 'Could not infer plate_id for all samples'
    # batch ids
    stats_df['batch_id'] = samples_df.set_index('sample_id').batch_id
    assert ~stats_df.plate_id.isna().any(), 'Could not infer plate_id for all samples'
    # final reads logscale
    stats_df['final_reads_log10'] = stats_df.nonchim.replace(0,0.1).apply(lambda x: np.log10(x))
    # filter rate 
    stats_df['filter_rate'] = stats_df.denoised / stats_df.DADA2_input

    return stats_df

def plot_target_balance(hap_df):
    '''
    read counts for sample-target combinations
    '''
    reads_per_sample = hap_df.groupby(['sample_id','target'])['reads'].sum().reset_index()
    # logscale, remove zeroes
    reads_per_sample['log10_reads'] = np.log10(reads_per_sample.reads.replace(0,np.nan))
    fig, ax = plt.subplots(1, 1, figsize=(20,4))
    sns.stripplot(data=reads_per_sample,
        x = 'target', y = 'log10_reads', hue = 'target', 
        alpha = .1, jitter = .3,
        ax = ax)
    return fig, ax

def plot_allele_balance(hap_df):
    '''
    allele balance vs log coverage
    '''
    is_het = (hap_df.reads_fraction < 1)
    het_frac = hap_df[is_het].reads_fraction
    het_reads_log = hap_df[is_het].reads.apply(lambda x: np.log10(x))
    het_plot = sns.jointplot(x=het_frac, y=het_reads_log, 
        kind="hex", height=8)
    het_plot.ax_joint.set_ylabel('reads (log10)')
    het_plot.ax_joint.set_ylabel('allele fraction')
    
    return het_plot

def qc(args):

    os.makedirs(args.outdir, exist_ok = True)
    
    hap_df = prep_hap(args.haplotypes)

    fig, ax = plot_target_balance(hap_df)
    fig.savefig(f'{args.outdir}/target_balance.png')

    het_plot = plot_allele_balance(hap_df)
    het_plot.savefig(f'{args.outdir}/allele_balance.png')

    samples_df = prep_samples(args.samples)
    stats_df = prep_stats(args.stats, hap_df, samples_df)

def main(cmd):
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('--haplotypes', help='Haplotypes tsv file', type=argparse.FileType('r'))
    parser.add_argument('--samples', help='Samples tsv file', type=argparse.FileType('r'))
    parser.add_argument('--stats', help='DADA2 stats tsv file', type=argparse.FileType('r'))
    parser.add_argument('--outdir', help='Output directory', default='qc')

    args = parser.parse_args(cmd)
    args.outdir=args.outdir.rstrip('/')
    qc(args)

if __name__ == '__main__':
    main(sys.argv[1:])