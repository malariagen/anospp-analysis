import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
import os
import argparse

from .util import *

DADA2_COLS = OrderedDict([
    ('input','removed by filterAndTrim'), 
    ('filtered','removed by denoising'),
    ('denoised','removed by merging'), 
    ('merged','removde by rmchimera'), 
    ('nonchim','removed by post-filtering'),
    ('final','retained') 

])

def plot_target_balance(hap_df):

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
    
    logging.info('plotting allele balance and coverage')

    is_het = (hap_df.reads_fraction < 1)
    het_frac = hap_df[is_het].reads_fraction
    het_reads_log = hap_df[is_het].reads.apply(lambda x: np.log10(x))
    het_plot = sns.jointplot(x=het_frac, y=het_reads_log, 
        kind="hex", height=8)
    het_plot.ax_joint.set_ylabel('reads (log10)')
    het_plot.ax_joint.set_ylabel('allele fraction')
    
    return het_plot

def plot_sample_target_heatmap(hap_df, samples_df, col):

    logging.info(f'plotting sample-target heatmap for {col}')

    if col not in ('total_reads','nalleles'):
        raise ValueError(f'sample target heatmap for {col} not implemented')

    st_df = hap_df.pivot_table(
        values=col, 
        index='target', 
        columns='sample_id', 
        aggfunc=np.max).fillna(0)
    
    if col == 'total_reads':
        st_df = np.log10(st_df.astype(float).replace(0, .1))
    
    plates = samples_df['plate_id'].unique()
    nplates = len(plates)

    fig, axs = plt.subplots(nplates,1,figsize=(22, 15 * nplates))

    for plate, ax in zip(plates, axs):
        plate_samples = samples_df.loc[samples_df['plate_id'] == plate, 'sample_id']
        plot_df = st_df.reindex(columns=plate_samples, fill_value=0)
        max_int = plot_df.max().max().astype(int) + 1
        sns.heatmap(plot_df, 
                    cmap='coolwarm', 
                    center=2, 
                    linewidths=.5, 
                    linecolor='silver', 
                    ax=ax, 
                    cbar_kws={'ticks':range(max_int)})
    plt.tight_layout()

    return fig, axs

def plot_sample_filtering(sample_stats_df, samples_df, dada2_cols=DADA2_COLS):
    
    logging.info('plotting per-sample filtering barplots')
    
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

def plot_plate_stats(comb_stats_df, lims_plate=False):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'

    logging.info('plotting plate stats')
    
    fig, axs = plt.subplots(3,1, figsize=(10,15))
    sns.stripplot(data=comb_stats_df,
                y='final_log10',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[0])
    # 1000 reads cutoff
    axs[0].axhline(3, c='silver')
    axs[0].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='targets_recovered',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[1])
    # 30 targets cutoff
    axs[1].axhline(30, c='silver')
    axs[1].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='filter_rate',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[2])
    # 50% filtering cutoff
    axs[2].axhline(.5, c='silver')
    for ax in axs:
        ax.get_legend().remove()
    plt.xticks(rotation=90)

    return fig, axs

def plot_plate_summaries(comb_stats_df, lims_plate=False):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'

    logging.info(f'plotting success summaries by {plate_col}')
    
    # success rate definition
    comb_stats_df['over 1000 final reads'] = comb_stats_df.denoised > 1000
    comb_stats_df['over 30 targets'] = comb_stats_df.targets_recovered > 30
    comb_stats_df['over 50% reads retained'] = comb_stats_df.filter_rate > .5

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()

    sum_df = comb_stats_df.groupby(plate_col)[['over 1000 final reads', 'over 30 targets','over 50% reads retained']].sum()
    y = comb_stats_df.groupby(plate_col)['over 1000 final reads'].count()
    sum_df = sum_df.divide(y, axis=0).reindex(plates)

    fig, ax = plt.subplots(1,1,figsize=(nplates * .5 + 2, 4))
    sns.heatmap(sum_df.T, annot=True, ax=ax, vmax=1, vmin=0)
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax

def plot_sample_success(comb_stats_df):

    logging.info('plotting sample success')

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    for col, ax in zip(('final_log10','filter_rate'), axs):
        sns.scatterplot(data=comb_stats_df,
                x='mosq_targets_recovered',
                y=col,
                hue='plate_id',
                alpha=.5, 
                ax=ax)
        ax.axvline(30, alpha=.5)
    axs[0].axhline(3, alpha=.5)
    axs[1].axhline(.5, alpha=.5)
    axs[1].get_legend().remove()

    return fig, axs

def plot_plate_heatmap(comb_stats_df, col, lims_plate=False, **heatmap_kwargs):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'
    well_col = 'lims_well_id' if lims_plate else 'well_id'
    plot_width = 12 if lims_plate else 8
    plot_height = 8 if lims_plate else 6

    logging.info(f'plotting heatmap for {col} by {plate_col}')

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()

    comb_stats_df['row'] = comb_stats_df[well_col].str.slice(0,1)
    comb_stats_df['col'] = comb_stats_df[well_col].str.slice(1).astype(int)

    fig, axs = plt.subplots(nplates,1,figsize=(plot_width, plot_height*nplates))
    for plate, ax in zip(plates, axs.flatten()):
        pdf = comb_stats_df[comb_stats_df[plate_col] == plate]
        hdf = pdf.pivot(index='row', columns='col', values=col)
        sns.heatmap(hdf, ax=ax, **heatmap_kwargs)
        if lims_plate:
            ax.hlines([i * 2 for i in range(9)],0,24,colors='k')
            ax.vlines([j * 2 for j in range(13)],0,16,colors='k')
        title = f'{plate} {col}'
        ax.set_title(title)
    plt.tight_layout()

    return fig, axs

def qc(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok = True)
    
    logging.info('ANOSPP QC data import started')

    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.samples)
    stats_df = prep_stats(args.stats)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)

    logging.info('ANOSPP QC plotting started')

    fig, _ = plot_target_balance(hap_df)
    fig.savefig(f'{args.outdir}/target_balance.png')

    fig = plot_allele_balance(hap_df)
    fig.savefig(f'{args.outdir}/allele_balance.png')

    for col in ('nalleles','total_reads'):
        fig, _ = plot_sample_target_heatmap(hap_df, samples_df, col=col)
        fig.savefig(f'{args.outdir}/sample_target_{col}.png')

    fig, _ = plot_sample_filtering(stats_df, samples_df)
    fig.savefig(f'{args.outdir}/filter_per_sample.png')

    fig, _ = plot_plate_stats(comb_stats_df, lims_plate=False)
    fig.savefig(f'{args.outdir}/plate_stats.png')

    fig, _ = plot_plate_summaries(comb_stats_df, lims_plate=False)
    fig.savefig(f'{args.outdir}/plate_summaries.png')

    fig, _ = plot_plate_summaries(comb_stats_df, lims_plate=True)
    fig.savefig(f'{args.outdir}/lims_plate_summaries.png')

    fig, _ = plot_sample_success(comb_stats_df)
    fig.savefig(f'{args.outdir}/sample_success.png')

    heatmap_kwargs = {
        'center':None,
        'annot':True,
        'cmap':'coolwarm',
        'fmt':'.2g'
    }
    for col in ('input_log10', 
                'final_log10',
                'filter_rate',
                'mosq_targets_recovered',
                'P1_log10_reads',
                'P2_log10_reads'):
        for lims_plate in (True, False):
            if col == 'mosq_targets_recovered':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 62
            elif col == 'filter_rate':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 1                
            fig, _ = plot_plate_heatmap(
                comb_stats_df,
                col=col,
                lims_plate=lims_plate,
                **heatmap_kwargs)
            if lims_plate:
                plate_hm_fn  = f'{args.outdir}/lims_plate_hm_{col}.png' 
            else:
                plate_hm_fn  = f'{args.outdir}/plate_hm_{col}.png' 
            fig.savefig(plate_hm_fn)
            plt.close(fig)

    logging.info('ANOSPP data QC ended')

def main():
    
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('--haplotypes', help='Haplotypes tsv file')
    parser.add_argument('--samples', help='Samples tsv file')
    parser.add_argument('--stats', help='DADA2 stats tsv file')
    parser.add_argument('--outdir', help='Output directory', default='qc')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    qc(args)

if __name__ == '__main__':
    main()