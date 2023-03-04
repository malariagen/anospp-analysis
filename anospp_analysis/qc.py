import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import OrderedDict
import sys
import os
import argparse

from utils import *

DADA2_COLS = OrderedDict([
    ('input','removed by filterAndTrim'), 
    ('filtered','removed by denoising'),
    ('denoised','removed by merging'), 
    ('merged','removde by rmchimera'), 
    ('nonchim','removed by post-filtering'),
    ('final','retained') 

])

def plot_target_balance(hap_df):
    '''
    read counts for sample-target combinations
    '''

    logging.info('plotting targets balance')
    reads_per_sample = hap_df.groupby(['sample_id','target'])['reads'].sum().reset_index()
    # logscale, remove zeroes
    reads_per_sample['log10_reads'] = reads_per_sample.reads.replace(0,np.nan).apply(lambda x: np.log10(x))
    fig, ax = plt.subplots(1, 1, figsize=(20,4))
    sns.stripplot(data=reads_per_sample,
        x = 'target', y = 'log10_reads', hue = 'target', 
        alpha = .1, jitter = .3,
        ax = ax)
    ax.set_ylabel('reads (log10)')
    ax.set_xlabel('target')
    return fig, ax

def plot_allele_balance(hap_df):
    '''
    allele balance vs log coverage
    '''

    logging.info('plotting allele balance and coverage')
    is_het = (hap_df.reads_fraction < 1)
    het_frac = hap_df[is_het].reads_fraction
    het_reads_log = hap_df[is_het].reads.apply(lambda x: np.log10(x))
    het_plot = sns.jointplot(x=het_frac, y=het_reads_log, 
        kind="hex", height=8)
    het_plot.ax_joint.set_ylabel('reads (log10)')
    het_plot.ax_joint.set_ylabel('allele fraction')
    
    return het_plot

def plot_sample_filtering(sample_stats_df, samples_df, dada2_cols=DADA2_COLS):
    '''
    Per-sample DADA2 filtering barplot
    '''
    
    logging.info('plotting per-sample filtering barplots')
    
    # TODO sum stats across targets and log10 transform
    plates = samples_df.plate_id.unique()
    nplates = len(plates)
    fig, axs = plt.subplots(nplates,1,figsize=(20, 4 * nplates))
    for plate, ax in zip(plates, axs):
        plate_samples = samples_df.loc[samples_df.plate_id == plate, 'sample_id']
        plot_df = sample_stats_df[sample_stats_df['sample_id'].isin(plate_samples)]
        for i, col in enumerate(dada2_cols.keys()):
            sns.barplot(x='sample_id',  y=f'{col}_log10', data=plot_df, 
                        color=sns.color_palette()[i], ax=ax, label=dada2_cols[col])
        ax.set_xlabel('sample_id')
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('reads (log10)')
        ax.legend(loc='upper left')
    plt.tight_layout()

    return fig, axs

def plot_plate_stats(comb_stats_df):

    logging.info('plotting plate stats')
    
    fig, axs = plt.subplots(3,1, figsize=(10,15))
    sns.stripplot(data=comb_stats_df,
                y='final_log10',
                x='plate_id',
                hue='plate_id',
                alpha=.3,
                jitter=.35,
                ax=axs[0])
    # 1000 reads cutoff
    axs[0].axhline(3, c='silver')
    axs[0].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='targets_recovered',
                x='plate_id',
                hue='plate_id',
                alpha=.3,
                jitter=.35,
                ax=axs[1])
    # 30 targets cutoff
    axs[1].axhline(30, c='silver')
    axs[1].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='filter_rate',
                x='plate_id',
                hue='plate_id',
                alpha=.3,
                jitter=.35,
                ax=axs[2])
    # 50% filtering cutoff
    axs[2].axhline(.5, c='silver')
    for ax in axs:
        ax.get_legend().remove()
    plt.xticks(rotation=90)

    return fig, axs

def qc(args):

    setup_logging(verbose=True)
    logging.info('ANOSPP data QC started')
    os.makedirs(args.outdir, exist_ok = True)
    
    hap_df = prep_hap(args.haplotypes)

    target_balance_fig, _ = plot_target_balance(hap_df)
    target_balance_fig.savefig(f'{args.outdir}/target_balance.png')

    het_plot = plot_allele_balance(hap_df)
    het_plot.savefig(f'{args.outdir}/allele_balance.png')

    samples_df = prep_samples(args.samples)
    stats_df = prep_stats(args.stats)

    # sample_filtering_fig, _ = plot_sample_filtering(stats_df, samples_df)
    # sample_filtering_fig.savefig(f'{args.outdir}/filter_per_sample.png')

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)

    plate_stats_fig, _ = plot_plate_stats(comb_stats_df)
    plate_stats_fig.savefig(f'{args.outdir}/plate_stats.png')

    logging.info('ANOSPP data QC ended')

def main(cmd):
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('--haplotypes', help='Haplotypes tsv file')
    parser.add_argument('--samples', help='Samples tsv file')
    parser.add_argument('--stats', help='DADA2 stats tsv file')
    parser.add_argument('--outdir', help='Output directory', default='qc')

    args = parser.parse_args(cmd)
    args.outdir=args.outdir.rstrip('/')
    qc(args)

if __name__ == '__main__':
    main(sys.argv[1:])