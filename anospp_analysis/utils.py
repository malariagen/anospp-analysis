import pandas as pd
import numpy as np
import logging

MOSQ_TARGETS = [str(i) for i in range(62)]
PLASM_TARGETS = ['P1','P2']
TARGETS = MOSQ_TARGETS + PLASM_TARGETS
CUTADAPT_TARGETS = TARGETS + ['unknown']

def setup_logging(verbose=False):
    try: 
        del logging.root.handlers[:]
    except:
        pass
    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

def well_id_mapper():
    '''
    Yields mapping of tag_index 1,2...96 
    to well id A1,B1...H12.
    N.B. subsequent plates can be mapped using
    tag_index % 96, thus last well is inserted at 0
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
    tag_index % 384, thus last well is inserted at 0
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

def seqid_generator(hap_df):

    seqids = dict()
    for tgt, group in hap_df.groupby(['target']):
        for (i, cons) in enumerate(group['consensus'].unique()):
            seqids[tgt + cons] = '{}-{}'.format(tgt, i)
    hap_df['seqid'] = (hap_df.target + hap_df.consensus).replace(seqids)

    return hap_df

def prep_hap(hap_fn):
    '''
    load haplotypes table
    '''

    logging.info(f'preparing haplotypes table from {hap_fn}')

    hap_df = pd.read_csv(hap_fn, sep='\t')

    # compatibility with old style haplotype column names
    hap_df.rename(columns=({
        's_Sample':'sample_id',
        'frac_reads':'reads_fraction'
        }), 
    inplace=True)

    for col in ('sample_id',
                'target',
                'consensus',
                'reads'):
        assert col in hap_df.columns, 'hap column {col} not found'

    if 'reads_log10' not in hap_df.columns:
        hap_df['reads_log10'] = hap_df['reads'].apply(lambda x: np.log10(x))

    if 'total_reads' not in hap_df.columns:
        hap_df['total_reads'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['reads'].transform('sum')

    if 'reads_fraction' not in hap_df.columns:
        hap_df['reads_fraction'] = hap_df['reads'] / hap_df['total_reads']

    if 'nalleles' not in hap_df.columns:
        hap_df['nalleles'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['consensus'].transform('nunique')
        
    hap_df = seqid_generator(hap_df)

    return hap_df

def prep_samples(samples_fn):
    '''
    load sample manifest used for anospp pipeline
    '''

    logging.info(f'preparing sample manifest from {samples_fn}')

    # allow reading from tsv (new style) or csv (old style)
    if samples_fn.endswith('csv'):
        samples_df = pd.read_csv(samples_fn, sep=',')
    elif samples_fn.endswith('tsv'):
        samples_df = pd.read_csv(samples_fn, sep='\t')
    else:
        raise ValueError(f'Expected {samples_fn} to be in either tsv or csv format')

    # compatibility with old style samples column names
    samples_df.rename(columns=({
        'Source_sample':'sample_id',
        # 'sample':'sample_id',
        'Run':'run_id',
        'Lane':'lane_index',
        'Tag':'tag_index',
        'Replicate':'replicate_id'
        }), 
    inplace=True)

    for col in ('sample_id',
                'run_id',
                'lane_index',
                'tag_index'):
        assert col in samples_df.columns, 'samples column {col} not found'

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
    if ('id_library_lims' in samples_df.columns and
        samples_df.id_library_lims.str.contains(':').all()):
            samples_df[['lims_plate_id','lims_well_id']] = samples_df.id_library_lims.str.split(':', 
                                                                                                n = 1, 
                                                                                                expand=True)
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
    
    For legacy stats table, summarise across targets
    '''

    logging.info(f'preparing DADA2 statistics from {stats_fn}')

    stats_df = pd.read_csv(stats_fn, sep='\t')
    # compatibility with legacy samples column names
    stats_df.rename(columns={
        's_Sample':'sample_id'
        },
        inplace=True)
    
    for col in ('sample_id',
                'input',
                'filtered',
                'denoisedF',
                'denoisedR',
                'merged',
                'nonchim'):
        assert col in stats_df.columns, 'stats column {col} not found'

    # denoising happens for F and R reads independently, we take minimum of those 
    # as an estimate for retained read count
    stats_df['denoised'] = stats_df[['denoisedF','denoisedR']].min(axis=1)
    # legacy stats calculated separately for each target, merging
    if 'target' in stats_df.columns:
        stats_df = stats_df.groupby('sample_id').sum(numeric_only=True).reset_index()
    # add final read counts for comatibility with legacy pipeline
    # that had a post-filtering step
    if 'final' not in stats_df.columns:
        stats_df['final'] = stats_df['nonchim']
    # logscale read counts, placeholder value for zero - -1
    for col in stats_df.columns.drop(['sample_id']):
        stats_df[f'{col}_log10'] = stats_df[col].replace(0,0.1).apply(lambda x: np.log10(x))
    # filter rate 
    stats_df['filter_rate'] = stats_df['nonchim'] / stats_df['input']
    
    return stats_df

def combine_stats(stats_df, hap_df, samples_df):
    '''
    Combined per-sample statistics
    '''

    logging.info('preparing combined stats')

    # some samples can be missing from stats due to DADA2 filtering
    if set(stats_df.sample_id) - set(samples_df.sample_id) != set():
        logging.error('sample_id mismatch between samples and stats, QC results will be compromised')
    # some samples can be missing from haps due to DADA2 & post-processing filtering
    elif set(hap_df.sample_id) - set(samples_df.sample_id) != set():
        logging.error('sample_id mismatch between haps and samples, QC results will be compromised')

    comb_stats_df = pd.merge(stats_df, samples_df, on='sample_id', how='inner')
    comb_stats_df.set_index('sample_id', inplace=True)
    comb_stats_df['targets_recovered'] = hap_df.groupby('sample_id') \
        ['target'].nunique()
    comb_stats_df['targets_recovered'] = comb_stats_df['targets_recovered'].fillna(0)
    comb_stats_df['mosq_targets_recovered'] = hap_df[hap_df.target.isin(MOSQ_TARGETS)] \
        .groupby('sample_id')['target'].nunique()
    comb_stats_df['mosq_targets_recovered'] = comb_stats_df['mosq_targets_recovered'].fillna(0)
    comb_stats_df['mosq_reads'] = hap_df[hap_df.target.isin(MOSQ_TARGETS)] \
        .groupby('sample_id')['reads'].sum()
    comb_stats_df['mosq_reads'] = comb_stats_df['mosq_reads'].fillna(0)
    comb_stats_df['mosq_log10_reads'] = comb_stats_df['mosq_reads'] \
        .replace(0,0.1).apply(lambda x: np.log10(x))
    for pt in PLASM_TARGETS:
        comb_stats_df[f'{pt}_reads'] = hap_df[hap_df.target == pt] \
            .groupby('sample_id')['reads'].sum()
        comb_stats_df[f'{pt}_reads'] = comb_stats_df[f'{pt}_reads'].fillna(0)
        comb_stats_df[f'{pt}_log10_reads'] = comb_stats_df[f'{pt}_reads'] \
            .replace(0,0.1).apply(lambda x: np.log10(x))
    comb_stats_df.reset_index(inplace=True)
        
    return comb_stats_df