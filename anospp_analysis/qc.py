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

TARGETS = [str(i) for i in range(62)] + ['P1','P2']
CUTADAPT_TARGETS = TARGETS + ['unknown']

def well_id_mapper():
    '''
    Yields mapping of tag_index 1,2...96 
    to well id A1,B1...H12.
    N.B. subsequent plates can be mapped using
    tag_index % 96
    '''

    well_ids = dict()
    tag = 1
    for col in range(1,13):
        for row in 'ABCDEFGH':
            well_ids[tag] = f'{row}{col}'
            tag += 1

    # edge case
    well_ids[0] = 'H12'
            
    return well_ids

def lims_well_id_mapper():
    '''
    Yields mapping of tag_index 1,2...384 
    to well id A1,C1...P24
    4 96-well plates order in quadrants is
    1 2
    3 4
    N.B. subsequent plates can be mapped using
    tag_index % 384
    '''
    lims_well_ids = dict()
    tag = 1
    # upper left quadrant
    for col in range(1,13):
        for row in 'ACEGIKMO':
            lims_well_ids[tag] = f'{row}{col * 2 - 1}'
            tag += 1
    # upper right quadrant
    for col in range(1,13):
        for row in 'ACEGIKMO':
            lims_well_ids[tag] = f'{row}{col * 2}'
            tag += 1
    # lower left quadrant
    for col in range(1,13):
        for row in 'BDFHJLNP':
            lims_well_ids[tag] = f'{row}{col * 2 - 1}'
            tag += 1
    # lower right quadrant
    for col in range(1,13):
        for row in 'BDFHJLNP':
            lims_well_ids[tag] = f'{row}{col * 2}'
            tag += 1

    # edge case
    lims_well_ids[0] = 'P24'

    return lims_well_ids

def prep_hap(hap_fn):
    '''
    load haplotypes table
    '''

    hap_df = pd.read_csv(hap_fn, sep='\t')

    # compatibility with old style haplotype column names
    hap_df.rename(columns=({
        's_Sample':'sample_id',
        'frac_reads':'reads_fraction'
        }), 
    inplace=True)

    return hap_df

def prep_samples(samples_fn):
    '''
    load sample manifest used for anospp pipeline
    '''

    # temp - allow reading from tsv or csv
    # TODO converge to tsv
    if str(samples_fn).endswith('csv'):
        samples_df = pd.read_csv(samples_fn, sep=',')
    elif str(samples_fn).endswith('tsv'):
        samples_df = pd.read_csv(samples_fn, sep='\t')
    else:
        raise ValueError(f'Expected {samples_fn} to be in either tsv or csv format')

    # compatibility with old style samples column names
    samples_df.rename(columns=({
        'Source_sample':'sample_id',
        'Run':'run_id',
        'Lane':'lane_index',
        'Tag':'tag_index',
        'Replicate':'replicate_id'
        }), 
    inplace=True)
    
    samples_df.set_index('sample_id', inplace=True)

    # plate ids
    if 'plate_id' in samples_df.columns:
        samples_df['plate_id'] = samples_df.plate_id
    else:
        samples_df['plate_id'] = samples_df.apply(lambda r: f'p_{r.run_id}_{(r.tag_index - 1) // 96 + 1}',
            axis=1)
    if 'well_id' in samples_df.columns:
        samples_df['well_id'] = samples_df.well_id
    else:
        samples_df['well_id'] = (samples_df.tag_index % 96).replace(well_id_mapper())
    assert ~samples_df.plate_id.isna().any(), 'Could not infer plate_id for all samples'
    assert ~samples_df.well_id.isna().any(), 'Could not infer well_id for all samples'
    assert samples_df.well_id.isin(well_id_mapper().values()).all(), 'Found well_id outside A1...H12'
    # lims plate ids
    if samples_df.id_library_lims.str.contains(':').all():
        samples_df[['lims_plate_id','lims_well_id']] = samples_df.id_library_lims.str.split(':',n=1,expand=True)
    else:
        samples_df['lims_plate_id'] = samples_df.apply(lambda r: f'lp_{r.run_id}_{(r.tag_index - 1) // 384 + 1}',
            axis=1)
        samples_df['lims_well_id'] = (samples_df.tag_index % 384).replace(lims_well_id_mapper())
    assert ~samples_df.lims_plate_id.isna().any(), 'Could not infer plate_id for all samples'
    assert ~samples_df.lims_well_id.isna().any(), 'Could not infer well_id for all samples'
    assert samples_df.lims_well_id.isin(lims_well_id_mapper().values()).all(), 'Found well_id outside A1...H12'

    return samples_df

def prep_stats(stats_fn):
    '''
    load DADA2 stats table
    '''

    stats_df = pd.read_csv(stats_fn, sep='\t', index_col=0)
    # compatibility with old style samples column names
    stats_df.rename(columns={
        's_Sample':'sample_id'
    },
    inplace=True)
    # denoising happens for F and R reads independently, we take minimum of those 
    # as an estimate for retained read count
    stats_df['denoised'] = stats_df[['denoisedF','denoisedR']].min(axis=1)
    
    return stats_df

def combine_stats(stats_df, hap_df, samples_df):
    # # fix samples order
    # stats_df = stats_df.reindex(samples_df['sample_id'])
    # targets per sample - assume same samples as in haplotypes
    stats_df['targets_recovered'] = hap_df.groupby('sample_id').target.nunique().fillna(0)
    # assert ~stats_df.targets_recovered.isna().any(), 'Could not calculate targets_recovered for all samples'
    

    # batch ids
    # stats_df['batch_id'] = samples_df.batch_id
    # assert ~stats_df.plate_id.isna().any(), 'Could not infer plate_id for all samples'
    # # final reads logscale, placeholder value for zero - -1
    # stats_df['final_reads_log10'] = stats_df.nonchim.replace(0,0.1).apply(lambda x: np.log10(x))
    # # filter rate 
    # stats_df['filter_rate'] = stats_df.denoised / stats_df.DADA2_input

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

    # fig, ax = plot_target_balance(hap_df)
    # fig.savefig(f'{args.outdir}/target_balance.png')

    # het_plot = plot_allele_balance(hap_df)
    # het_plot.savefig(f'{args.outdir}/allele_balance.png')

    samples_df = prep_samples(args.samples)
    stats_df = prep_stats(args.stats)

    comb_df = combine_stats(stats_df, hap_df, samples_df)

def main(cmd):
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('--haplotypes', help='Haplotypes tsv file') #, type=argparse.FileType('r'))
    parser.add_argument('--samples', help='Samples tsv file') #, type=argparse.FileType('r'))
    parser.add_argument('--stats', help='DADA2 stats tsv file') #, type=argparse.FileType('r'))
    parser.add_argument('--outdir', help='Output directory', default='qc')

    args = parser.parse_args(cmd)
    args.outdir=args.outdir.rstrip('/')
    qc(args)

if __name__ == '__main__':
    main(sys.argv[1:])