import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse

from .util import *

def plot_target_balance(hap_df):

    logging.info('plotting targets balance')

    reads_per_sample = hap_df.groupby(['sample_id','target'])['reads'].sum().reset_index()
    # logscale, remove zeroes
    reads_per_sample['log10_reads'] = reads_per_sample.reads.replace(0,np.nan).apply(lambda x: np.log10(x))
    
    figsize = (hap_df['target'].nunique() * 0.3, 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.stripplot(data=reads_per_sample,
        x = 'target', y = 'log10_reads', hue = 'target', 
        alpha = .1, jitter = .3,
        ax = ax)
    ax.get_legend().remove()
    ax.set_ylabel('reads (log10)')
    ax.set_xlabel('target')
    ax.axhline(1, c='silver', alpha=.5)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()

    return fig, ax

def plot_allele_balance(hap_df):
    
    logging.info('plotting allele balance and coverage')

    is_het = (hap_df.reads_fraction < 1)
    het_frac = hap_df[is_het].reads_fraction
    het_reads_log = hap_df[is_het].reads.apply(lambda x: np.log10(x))
    het_plot = sns.jointplot(x=het_frac, y=het_reads_log, 
        kind="hist", height=8)
    het_plot.ax_joint.set_ylabel('reads (log10)')
    het_plot.ax_joint.set_xlabel('allele fraction')
    het_plot.ax_joint.axhline(1, c='silver', alpha=.5)
    het_plot.ax_joint.axvline(0.1, c='silver', alpha=.5)
    
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
        ax.set_title(plate)
    plt.tight_layout()

    return fig, axs

def plot_sample_filtering(comb_stats_df):
    
    logging.info('plotting per-sample filtering barplots')

    # comb_stats_df colname : legend label
    dada2_cols = OrderedDict([
        ('input','removed by filterAndTrim'), 
        ('filtered','removed by denoising'),
        ('denoised','removed by merging'), 
        ('merged','removde by rmchimera'), 
        ('nonchim','removed by post-filtering'),
        ('final','retained')
        ])
    
    plates = comb_stats_df.plate_id.unique()
    nplates = len(plates)
    fig, axs = plt.subplots(nplates,1,figsize=(20, 4 * nplates))
    for plate, ax in zip(plates, axs):
        plot_df = comb_stats_df[comb_stats_df.plate_id == plate]
        for i, col in enumerate(dada2_cols.keys()):
            sns.barplot(x='sample_id',  y=f'{col}_log10', data=plot_df, 
                        color=sns.color_palette()[i], ax=ax, label=dada2_cols[col])
        ax.set_xlabel('sample_id')
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('reads (log10)')
        ax.legend(loc='upper left')
        ax.set_title(plate)
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
    axs[0].axhline(3, c='silver', alpha=.5)
    axs[0].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='targets_recovered',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[1])
    # 30 targets cutoff
    axs[1].axhline(30, c='silver', alpha=.5)
    axs[1].set_xticklabels([])
    sns.stripplot(data=comb_stats_df,
                y='filter_rate',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[2])
    # 50% filtering cutoff
    axs[2].axhline(.5, c='silver', alpha=.5)
    for ax in axs:
        ax.get_legend().remove()
    plt.xticks(rotation=90)
    plt.tight_layout()

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

def plot_sample_success(comb_stats_df, anospp=True):

    logging.info('plotting sample success')

    xcol = 'raw_mosq_targets_recovered' if anospp else 'targets_recovered'

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    for ycol, ax in zip(('final_log10','filter_rate'), axs):
        sns.scatterplot(data=comb_stats_df,
                x=xcol,
                y=ycol,
                hue='plate_id',
                alpha=.5, 
                ax=ax)
        ax.axvline(30, c='silver', alpha=.5)
    axs[0].axhline(3, c='silver', alpha=.5)
    axs[1].axhline(.5, c='silver', alpha=.5)
    axs[1].get_legend().remove()

    return fig, axs

def plot_plasm_balance(comb_stats_df):

    logging.info('plotting Plasmodium read balance')

    max_p_log = max(comb_stats_df.P1_log10_reads.max(), 
                    comb_stats_df.P2_log10_reads.max())

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.scatterplot(data=comb_stats_df,
                    x='P1_log10_reads',
                    y='P2_log10_reads',
                    hue='plate_id',
                    alpha=.5, 
                    ax=ax)
    ax.axhline(1, c='silver', alpha=.5)
    ax.axvline(1, c='silver', alpha=.5)
    ax.plot([-1, max_p_log], [-1, max_p_log], color='silver', linestyle='dashed', alpha=.5)

    return fig, ax

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
    if nplates == 1:
        axs = np.array([axs])
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

    samples_df = prep_samples(args.manifest)
    hap_df = prep_hap(args.haplotypes)
    stats_df = prep_stats(args.stats)
    
    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)


    logging.info('saving combined stats')

    comb_stats_df.to_csv(f'{args.outdir}/combined_stats.tsv', sep='\t', index=False)

    logging.info('starting plotting QC')
    
    if hap_df['target'].isin(CUTADAPT_TARGETS).all():
        anospp = True
        logging.info('only ANOSPP targets detected, plotting ANOSPP QC')
    else:
        anospp = False
        logging.warning('non-ANOSPP targets detected, plotting generic QC')

    fig, _ = plot_target_balance(hap_df)
    fig.savefig(f'{args.outdir}/target_balance.png')

    fig = plot_allele_balance(hap_df)
    fig.savefig(f'{args.outdir}/allele_balance.png')

    for col in ('nalleles','total_reads'):
        fig, _ = plot_sample_target_heatmap(hap_df, samples_df, col=col)
        fig.savefig(f'{args.outdir}/sample_target_{col}.png')

    fig, _ = plot_sample_filtering(comb_stats_df)
    fig.savefig(f'{args.outdir}/filter_per_sample.png')

    fig, _ = plot_plate_stats(comb_stats_df, lims_plate=False)
    fig.savefig(f'{args.outdir}/plate_stats.png')

    fig, _ = plot_plate_summaries(comb_stats_df, lims_plate=False)
    fig.savefig(f'{args.outdir}/plate_summaries.png')

    fig, _ = plot_plate_summaries(comb_stats_df, lims_plate=True)
    fig.savefig(f'{args.outdir}/lims_plate_summaries.png')

    fig, _ = plot_sample_success(comb_stats_df, anospp=anospp)
    fig.savefig(f'{args.outdir}/sample_success.png')

    if anospp:
        fig, _ = plot_plasm_balance(comb_stats_df)
        fig.savefig(f'{args.outdir}/plasm_balance.png')

    heatmap_kwargs = {
        'center':None,
        'annot':True,
        'cmap':'coolwarm',
        'fmt':'.2g'
    }

    if anospp:
        heatmap_cols = ['input_log10', 
                'final_log10',
                'filter_rate',
                'raw_mosq_targets_recovered',
                'P1_log10_reads',
                'P2_log10_reads',
                'raw_multiallelic_mosq_targets']
    else:
        heatmap_cols = ['input_log10', 
                'final_log10',
                'filter_rate',
                'targets_recovered',
                'multiallelic_targets']

    for col in heatmap_cols:
        for lims_plate in (True, False):
            if col == 'raw_mosq_targets_recovered':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 62
            elif col == 'raw_multiallelic_mosq_targets':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            elif col == 'filter_rate':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 1
            elif 'log10' in col: # read counts
                heatmap_kwargs['vmin'] = -1
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            else:
                heatmap_kwargs['vmin'] = None
                heatmap_kwargs['vmax'] = None             
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

    logging.info('ANOSPP QC complete')

def main():
    
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='qc')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    qc(args)

if __name__ == '__main__':
    main()