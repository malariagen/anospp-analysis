import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import OrderedDict
import os
import argparse

from anospp_analysis.util import *

def plot_target_balance(hap_df):

    logging.info('plotting targets balance')

    reads_per_sample_target = hap_df.groupby(['sample_id','target'])['reads'].sum().reset_index()
    
    figsize = (hap_df['target'].nunique() * 0.3, 6)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.stripplot(data=reads_per_sample_target,
        x = 'target', y = 'reads', hue = 'target', 
        alpha = .1, jitter = .3, ax = ax)
    ax.get_legend().remove()
    ax.set_yscale('log')
    ax.set_ylabel('reads')
    ax.set_xlabel('target')
    ax.axhline(10, c='silver', alpha=.5)
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
        ('total_reads','removed as readthrough'),
        ('DADA2_input_reads','removed by filterAndTrim'), 
        ('DADA2_filtered_reads','removed by denoising'),
        ('DADA2_denoised_reads','removed by merging'), 
        ('DADA2_merged_reads','removde by rmchimera'), 
        ('DADA2_nonchim_reads','unassigned to amplicons'),
        # legacy post-filter disabled in prod
        # ('DADA2_final_reads','unassigned to amplicons'),
        ('deplexed_reads','Plasmodium reads'),
        ('raw_mosq_reads','mosquito reads')
        ])
    
    plates = comb_stats_df.plate_id.unique()
    nplates = len(plates)
    fig, axs = plt.subplots(nplates,1,figsize=(20, 4 * nplates))
    for plate, ax in zip(plates, axs):
        plot_df = comb_stats_df[comb_stats_df.plate_id == plate].copy()
        plot_df['well_id'] = well_ordering(plot_df['well_id'])
        plot_df.sort_values(by='well_id', inplace=True)
        for i, col in enumerate(dada2_cols.keys()):
            sns.barplot(x='sample_id', y=col, data=plot_df, 
                        color=sns.color_palette()[i], ax=ax,
                        label=dada2_cols[col])
        ax.set_xticklabels(plot_df['sample_name'])
        ax.set_xlabel('sample_name')
        ax.tick_params(axis='x', rotation=90)
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5, top=max(comb_stats_df['total_reads']))
        ax.set_ylabel('reads')
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        ax.set_title(plate)
    plt.tight_layout()

    return fig, axs

def plot_plate_stats(comb_stats_df, lims_plate=False):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'

    logging.info('plotting plate stats')
    
    fig, axs = plt.subplots(3,1, figsize=(10,15))
    sns.stripplot(data=comb_stats_df,
                y='deplexed_reads',
                x=plate_col,
                hue=plate_col,
                alpha=.3,
                jitter=.35,
                ax=axs[0])
    axs[0].set_yscale('log')
    # 1000 reads cutoff
    axs[0].axhline(1000, c='silver', alpha=.5)
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
                y='overall_filter_rate',
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
    comb_stats_df['over 1000 mosquito reads'] = comb_stats_df.raw_mosq_reads > 1000
    comb_stats_df['over 30 targets'] = comb_stats_df.targets_recovered > 30
    comb_stats_df['over 50% reads retained'] = comb_stats_df.overall_filter_rate > .5

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()

    sum_df = comb_stats_df.groupby(plate_col)[['over 1000 mosquito reads', 'over 30 targets','over 50% reads retained']].sum()
    y = comb_stats_df.groupby(plate_col)['over 1000 mosquito reads'].count()
    sum_df = sum_df.divide(y, axis=0).reindex(plates)

    fig, ax = plt.subplots(1,1,figsize=(nplates * .5 + 2.5, 4))
    sns.heatmap(sum_df.T, annot=True, ax=ax, vmax=1, vmin=0)
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax

def plot_sample_success(comb_stats_df, anospp=True):

    logging.info('plotting sample success')

    xcol = 'raw_mosq_targets_recovered' if anospp else 'targets_recovered'

    fig, axs = plt.subplots(1,2,figsize=(12,6))
    for ycol, ax in zip(('raw_mosq_reads', 'overall_filter_rate'), axs):
        sns.scatterplot(data=comb_stats_df,
                x=xcol,
                y=ycol,
                hue='plate_id',
                alpha=.5, 
                ax=ax)
        ax.axvline(30, c='silver', alpha=.5)
    axs[0].set_yscale('log')
    axs[0].axhline(1000, c='silver', alpha=.5)
    axs[1].axhline(.5, c='silver', alpha=.5)
    axs[1].get_legend().remove()

    return fig, axs

def plot_plasm_balance(comb_stats_df):

    logging.info('plotting Plasmodium read balance')

    max_p_log = max(comb_stats_df.P1_reads.max(), 
                    comb_stats_df.P2_reads.max())

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.scatterplot(data=comb_stats_df,
                    x='P1_reads',
                    y='P2_reads',
                    hue='plate_id',
                    alpha=.5, 
                    ax=ax)
    ax.axhline(10, c='silver', alpha=.5)
    ax.axvline(10, c='silver', alpha=.5)
    ax.plot([0.9, max_p_log], [0.9, max_p_log], color='silver', linestyle='dashed', alpha=.5)
    ax.set_yscale('log')
    ax.set_xscale('log')

    return fig, ax

def plot_plate_heatmap(comb_stats_df, col, lims_plate=False, **heatmap_kwargs):

    plate_col = 'lims_plate_id' if lims_plate else 'plate_id'
    well_col = 'lims_well_id' if lims_plate else 'well_id'
    plot_width = 14 if lims_plate else 9
    plot_height = 8 if lims_plate else 6

    logging.info(f'plotting heatmap for {col} by {plate_col}')

    plates = comb_stats_df[plate_col].unique()
    nplates = comb_stats_df[plate_col].nunique()

    comb_stats_df['row'] = comb_stats_df[well_col].str.slice(0, 1)
    comb_stats_df['col'] = comb_stats_df[well_col].str.slice(1).astype(int)

    fig, axs = plt.subplots(nplates, 1, figsize=(plot_width, plot_height * nplates))
    if nplates == 1:
        axs = np.array([axs])
    for plate, ax in zip(plates, axs.flatten()):
        pdf = comb_stats_df[comb_stats_df[plate_col] == plate]
        hdf = pdf.pivot(index='row', columns='col', values=col)
        # read counts adjustments
        if 'fmt' in heatmap_kwargs.keys():
            if heatmap_kwargs['fmt'] == '':
                # human formatted labels
                heatmap_kwargs['annot'] = hdf.applymap(human_format)
                # handling of zero counts
                hdf = hdf.replace(0, 0.1)
        sns.heatmap(hdf, ax=ax, **heatmap_kwargs)
        if lims_plate:
            ax.hlines([i * 2 for i in range(9)], 0, 24, colors='k')
            ax.vlines([j * 2 for j in range(13)], 0, 16, colors='k')
        title = f'{plate} {col}'
        ax.set_title(title)
    plt.tight_layout()

    return fig, axs

def qc(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok = True)
    
    logging.info('ANOSPP QC data import started')

    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)
    hap_df = prep_hap(args.haplotypes)
    
    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)

    logging.info('saving sample QC stats')

    comb_stats_df.to_csv(f'{args.outdir}/sample_qc_stats.tsv', sep='\t', index=False)
    
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

    # deactivated as unused
    # for col in ('nalleles','total_reads'):
    #     fig, _ = plot_sample_target_heatmap(hap_df, samples_df, col=col)
    #     fig.savefig(f'{args.outdir}/sample_target_{col}.png')

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

    if anospp:
        heatmap_cols = [
            'P1_reads',
            'P2_reads',
            'total_reads', 
            'deplexed_reads',
            'overall_filter_rate',
            'raw_mosq_targets_recovered',
            'raw_multiallelic_mosq_targets',
            'unassigned_haps'
            ]
    else:
        heatmap_cols = [
            'input', 
            'final',
            'overall_filter_rate',
            'targets_recovered',
            'multiallelic_targets'
            ]

    for col in heatmap_cols:
        for lims_plate in (True, False):
            # re-init heatmap args for each plot
            heatmap_kwargs = {
                    'annot':True,
                    'cmap':'coolwarm'
                }
            if col == 'raw_mosq_targets_recovered':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 62
            elif col == 'raw_multiallelic_mosq_targets':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            elif col == 'unassigned_haps':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = max(comb_stats_df[col])
            elif col == 'filter_rate':
                heatmap_kwargs['vmin'] = 0
                heatmap_kwargs['vmax'] = 1
                heatmap_kwargs['fmt'] = '.2f'
            # read counts
            else:
                # log-transform colour axis
                # vmin vmax set here
                heatmap_kwargs['norm'] = LogNorm(
                    vmin=0.1,
                    vmax=max(max(comb_stats_df[col]), 0.1))
                # auto-apply human_format to annot
                heatmap_kwargs['fmt'] = '' 
            

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