import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import itertools

from .util import *

def recompute_haplotype_coverage(hap_df):
    hap_df = hap_df.drop(["total_reads", "reads_fraction", "nalleles"], axis=1)

    hap_df['total_reads'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['reads'].transform('sum')

    hap_df['reads_fraction'] = hap_df['reads'] / hap_df['total_reads']

    hap_df['nalleles'] = hap_df.groupby(by=['sample_id', 'target']) \
            ['consensus'].transform('nunique')
    return hap_df

def prep_mosquito_haps(hap_df, rc_threshold, rf_threshold):
    '''
    prepare mosquito haplotype dataframe
    remove plasmodium haplotypes
    change targets to integers
    returns haplotype dataframe
    '''

    logging.info('preparing mosquito haplotypes')

    hap_df = hap_df.astype({'target': str})
    filtered_hap_df = hap_df[(hap_df.reads>=int(rc_threshold)) & (hap_df.reads_fraction>=float(rf_threshold))]
    if filtered_hap_df.shape[0] < hap_df.shape[0]:
        logging.warning(f'Removed {hap_df.shape[0] - filtered_hap_df.shape[0]} haplotypes \
                    with fewer than {rc_threshold} reads or lower fracion than {rf_threshold} of reads')
    mosq_hap_df = filtered_hap_df[filtered_hap_df.target.isin(MOSQ_TARGETS)]
    mosq_hap_df = mosq_hap_df.astype({'target': int})

    #recompute reads coverage after filtering
    mosq_hap_df = recompute_haplotype_coverage(mosq_hap_df)

    return mosq_hap_df

def prep_reference_index(reference_version, path_to_refversion):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_version}')

    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/haplotypes.tsv'), f'reference version {reference_version} at {reference_path} \
        does not contain required haplotypes.tsv file'
    ref_hap_df = pd.read_csv(f'{reference_path}/haplotypes.tsv', sep='\t')

    assert os.path.isfile(f'{reference_path}/multiallelism.tsv'), f'reference version {reference_version} at {reference_path} \
        does not contain required multiallelism.tsv file'
    true_multi_targets = pd.read_csv(f'{reference_path}/multiallelism.tsv', sep='\t')

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning('No version.txt file present for reference version {reference_version} at {reference_path}')
        version_name = 'unknown'

    allele_freqs = dict()
    colors = dict()
    for level in ['coarse', 'int', 'fine']:
        assert os.path.isfile(f'{reference_path}/allele_freq_{level}.npy'), f'reference version {reference_version} at {reference_path} \
            does not contain required allele_freq_{level}.npy file'
        af = np.load(f'{reference_path}/allele_freq_{level}.npy')
        allele_freqs[level] = af

        assert os.path.isfile(f'{reference_path}/sgp_{level}.txt'), f'reference version {reference_version} at {reference_path} \
            does not contain required sgp_{level}.txt file'
        sgp = []
        with open(f'{reference_path}/sgp_{level}.txt', 'r') as fn:
            for line in fn:
                sgp.append(line.strip())

        ref_hap_df[f'{level}_sgp'] = pd.Categorical(ref_hap_df[f'{level}_sgp'], sgp, ordered=True)

        if not os.path.isfile(f'{reference_path}/colors_{level}.npy'):
            logging.warning('No colors defined for plotting.')
        else:
            clr = np.load(f'{reference_path}/colors_{level}.npy')
            colors[level] = clr
        
    return(ref_hap_df, allele_freqs, true_multi_targets, colors, version_name)

def construct_kmer_dict(k):
    '''
    construct a k-mer dict
    associating each unique k-mer of length k with a unique non-negative integer <4**k
    bases are written in capitals
    returns a dictionary
    '''
    labels = []
    for i in itertools.product('ACGT', repeat=k):
        labels.append(''.join(i))
    kmerdict = dict(zip(labels, np.arange(4**k)))
    return kmerdict   

def parse_seqid(seqid):
    '''
    Parse seqid passed as a string 
    '''
    split_seqid = seqid.split('-')

    assert split_seqid[0] in MOSQ_TARGETS, f'seqid {seqid} refers to a non-mosquito target'
    try:
        parsed_seqid = (int(split_seqid[0]), int(split_seqid[1]))
    except:
        raise Exception(f'seqid {seqid} cannot be converted to integers')
    return parsed_seqid

def parse_seqids_series(seqids):
    '''
    Parse seqid or seqids passed as a pandas Series
    '''
    parsed_seqids = seqids.str.split('-', expand=True)
    parsed_seqids.columns = ['target', 'uidx']
    
    assert parsed_seqids['target'].isin(MOSQ_TARGETS).all(), 'Dataframe contains seqids referring to non-mosquito targets'
    try:
        parsed_seqids = parsed_seqids.astype(int)
    except:
        raise Exception('Dataframe contains seqids which cannot be converted to integers')
    return parsed_seqids

def construct_unique_kmer_table(mosq_hap_df, k):
    '''
    constructs a k-mer table of dimensions n_amp * maxallele * 4**k
    represting the k-mer table of each unique sequence in the dataframe
    maxallele is the maximum number of unique sequences per target
    n_amp is the number of mosquito targets
    input: k=k (length of k-mers), hap_df=dataframe with haplotypes
    output: k-mer table representing each unique haplotype in the hap dataframe
    '''

    logging.info('translating unique sequences to k-mers')

    kmerdict = construct_kmer_dict(k)
    #subset to unique haplotypes
    uniqueseq = mosq_hap_df[['seqid', 'consensus']].drop_duplicates()
    #determine shape of table by highest seqid
    parsed_seqids = parse_seqids_series(uniqueseq.seqid)
    maxid = parsed_seqids['uidx'].max()+1

    #initiate table to store kmer counts
    kmer_table = np.zeros((len(MOSQ_TARGETS), maxid, 4**k), dtype='int')
    #translate each unique haplotype to kmer counts
    for idx, seq in uniqueseq.iterrows():
        tgt = parsed_seqids.loc[idx,'target']
        id = parsed_seqids.loc[idx,'uidx']
        consensus = seq.consensus
        for i in np.arange(len(consensus)-(k-1)):
            kmer_table[tgt,id,kmerdict[consensus[i:i+k]]] += 1
    return kmer_table

def identify_error_seqs(mosq_hap_df, kmers, k, n_error_snps):
    '''
    Identify haplotypes resulting from sequencing/PCR errors
    Cannot distinguish between true heterozygote, contaminated homozygote and homozygote with error sequence
    So only look for errors for unique sequences at multiallelic targets
    '''

    logging.info('identifying haplotypes resulting from sequencing/PCR errors')
    #set the k-mer threshold for the number of snps allowed for errors
    threshold=n_error_snps*k+1
    seqid_size = mosq_hap_df.groupby('seqid').size()
    singleton_seqids = seqid_size[seqid_size==1].index
    error_candidates = mosq_hap_df.query('(seqid in @singleton_seqids) & (nalleles>2)')

    error_seqs = []
    for _, cand in error_candidates.iterrows():
        possible_sources = mosq_hap_df.query('(sample_id == @cand.sample_id) & (target == @cand.target) & \
                                             (not seqid in @error_seqs) & (seqid != @cand.seqid)')
        cand_parsed_seqid = parse_seqid(cand.seqid)
        possible_sources_parsed_seqids = parse_seqids_series(possible_sources.seqid)
        for possible_source in possible_sources_parsed_seqids['uidx']:
            abs_kmer_dist = np.abs(kmers[cand.target,cand_parsed_seqid[1],:] - kmers[cand.target,possible_source,:]).sum()
            if abs_kmer_dist<threshold:
                error_seqs.append(cand.seqid)
                break
    
    logging.info(f'identified {len(error_seqs)} error sequences')

    return error_seqs

def compute_kmer_distance(kmers, ref_kmers, tgt, qidx, refidx):
    '''
    compute k-mer distance between query kmer count
    and ref kmer count(s)
    returns absolute and normalised distance
    '''
    #identify k-mer mismatches
    diff = np.abs(ref_kmers[tgt, refidx, :] - kmers[tgt, qidx, :])
    total = np.sum(ref_kmers[tgt, refidx, :] + kmers[tgt, qidx, :], axis=1)
    #sum to get absolute distance
    dist = np.sum(diff, axis=1)
    #normalise
    norm_dist = dist / total

    return dist, norm_dist

def find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers):
    '''
    identify the nearest neighbours of the unique haplotypes in the reference dataset 
    '''
    #get idxs occupied for each target
    parsed_ref_seqids = parse_seqids_series(ref_hap_df.seqid.drop_duplicates())
    ref_idxs_per_target = parsed_ref_seqids.groupby('target')['uidx'].unique()

    nndict = dict()

    logging.info(f"identifying nearest neighbours for {non_error_hap_df.seqid.nunique()} unique haplotypes")
    
    #loop through unique haplotypes
    unique_seqids = non_error_hap_df.seqid.unique()
    for seqid in unique_seqids:
        tgt, qidx = parse_seqid(seqid)
        #compute distance between focal hap and all same target haps in ref index
        dist, norm_dist = compute_kmer_distance(kmers, ref_kmers, tgt, qidx, ref_idxs_per_target[tgt])
        #Find nearest neighbours
        nn_qidx = ref_idxs_per_target[tgt][norm_dist==norm_dist.min()]
        #include in dict
        nndict[seqid] = (nn_qidx, norm_dist.min())

    return nndict

def lookup_assignment_proportion(q_seqid, allele_frequencies, tgt, nndict, weight=1):

    #lookup nearest neighbour identifiers
    nnids = nndict[q_seqid][0]
    #lookup allele frequencies of nnids
    af_nn = allele_frequencies[tgt, nnids, :]
    #sum allele frequencies over nnids
    summed_af_nn = np.sum(af_nn, axis=0)
    #normalise proportion and weigth in number of alleles
    assignment_proportion = weight*summed_af_nn/np.sum(summed_af_nn)
    return assignment_proportion

def perform_nn_assignment_samples(hap_df, ref_hap_df, nndict, allele_freqs,\
                                  normalisation):
    '''
    The main NN assignment function
    it outputs three dataframes containing the assignment proportions to each species-group for the three levels
    '''
    #get samples with at least 10 targets
    test_samples = hap_df.groupby('sample_id').filter(lambda x: x['target'].nunique() >=10)['sample_id'].unique()

    logging.info(f'performing NN assignment for {len(test_samples)} samples with >=10 mosquito targets')

    #set up data-output as numpy arrays (will be made into dataframes later)
    results = dict({'coarse': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['coarse'].shape[2])), \
                    'int': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['int'].shape[2])), \
                    'fine': np.zeros((len(MOSQ_TARGETS), len(test_samples), allele_freqs['fine'].shape[2]))})

    for i, sample in enumerate(test_samples):
        #Restrict to targets amplified in focal sample
        targets = hap_df.loc[hap_df.sample_id == sample, 'target'].unique()
        
        #Per amplified target
        for tgt in targets:
            #Identify the unique IDs of the focal sample's haplotypes at target t
            alleles = hap_df.loc[(hap_df.sample_id == sample) & (hap_df.target == tgt), ['seqid', 'reads_fraction']]
            #for each haplotype
            for _, allele in alleles.iterrows():
                #for each assignment level
                for level in ['coarse', 'int', 'fine']:
                    if normalisation == 'n_alleles':
                    #lookup assignment proportion
                        assignment_proportion = lookup_assignment_proportion(allele.seqid, allele_freqs[level], \
                                                    tgt, nndict, 1/alleles.shape[0])
                    elif normalisation == 'reads_fraction':
                        assignment_proportion = lookup_assignment_proportion(allele.seqid, allele_freqs[level], \
                                                    tgt, nndict, allele.reads_fraction)
                    else:
                        logging.error("Not a valid allelism_normalisation method.")
                    #table[tgt,nsmp,:] += assignment_proportion
                    results[level][tgt,i,:] += assignment_proportion

    #print(f'shape of results arrays is {results_coarse.shape}, {results_int.shape} and {results_fine.shape}')
    results_df = dict()
    for level in ['coarse', 'int', 'fine']:
        #Average assignment results over amplified targets
        res = np.nansum(results[level], axis=0)/np.sum(np.nansum(results[level], axis=0), axis=1)[:,None]
        #Convert results to dataframes
        results_df[level] = pd.DataFrame(res, index=test_samples, columns=ref_hap_df[f'{level}_sgp'].cat.categories)  
    return results_df, test_samples

def recompute_sample_coverage(comb_stats_df, non_error_hap_df):
    '''
    recompute coverage stats after filtering and error removal
    '''
    logging.info('recompute coverage stats')
    comb_stats_df.set_index('sample_id', inplace=True)

    #recompute multiallelic calls after filtering and error removal
    comb_stats_df['multiallelic_mosq_targets'] = (non_error_hap_df.groupby('sample_id')['target'].value_counts() > 2 \
        ).groupby(level='sample_id').sum()
    comb_stats_df['multiallelic_mosq_targets'] = comb_stats_df['multiallelic_mosq_targets'].fillna(0)

    #recompute read counts after filtering and error removal
    comb_stats_df['mosq_reads'] = non_error_hap_df.groupby('sample_id')['reads'].sum()
    comb_stats_df['mosq_reads'] = comb_stats_df['mosq_reads'].fillna(0)

    #recompute targets recovered after filtering and error removal
    comb_stats_df['mosq_targets_recovered'] = non_error_hap_df.groupby('sample_id')['target'].nunique()
    comb_stats_df['mosq_targets_recovered'] = comb_stats_df['mosq_targets_recovered'].fillna(0)

    comb_stats_df.reset_index(inplace=True)

    return comb_stats_df

def estimate_contamination(comb_stats_df, non_error_hap_df, true_multi_targets, \
                           rc_med_threshold, ma_med_threshold, ma_hi_threshold):
    '''
    estimate contamination from read counts and multiallelic targets
    '''
    logging.info('estimating contamination risk')

    #Read in exceptions from true_multi_targets file
    for idx, item in true_multi_targets.iterrows():
        potentially_affected_samples = comb_stats_df.loc[comb_stats_df[f'res_{item.level}'] == item.sgp, 'sample_id']
        affected_samples = non_error_hap_df.query('sample_id in @potentially_affected_samples & target == @item.target') \
            .groupby('sample_id').filter(lambda x: x['seqid'].nunique() > 2 & \
                                         x['seqid'].nunique() < item.admissable_alleles)['sample_id'].unique()
        comb_stats_df.loc[comb_stats_df.sample_id.isin(affected_samples), 'multiallelic_mosq_targets'] -= 1

    comb_stats_df.loc[comb_stats_df.multiallelic_mosq_targets>int(ma_hi_threshold), 'contamination_risk'] = 'high'
    comb_stats_df.loc[((comb_stats_df.multiallelic_mosq_targets>int(ma_med_threshold)) & \
                       (comb_stats_df.multiallelic_mosq_targets<=int(ma_hi_threshold))) |\
        (comb_stats_df.mosq_reads<int(rc_med_threshold)), 'contamination_risk'] = 'medium'
    comb_stats_df.loc[comb_stats_df.contamination_risk.isnull(), 'contamination_risk'] = 'low'

    logging.info(f"Identified {(comb_stats_df.contamination_risk=='high').sum()} samples with high contamination risk, \
        \n {(comb_stats_df.contamination_risk=='medium').sum()} samples with medium contamination risk \
        \n and {(comb_stats_df.contamination_risk=='low').sum()} samples with low contamination risk")

    return comb_stats_df

def generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, results_df, \
                        true_multi_targets, nn_asgn_threshold, \
                        rc_med_threshold, ma_med_threshold, ma_hi_threshold):

    logging.info('generating NN calls from assignment info')

    #Account for filtering and error removal
    comb_stats_df = recompute_sample_coverage(comb_stats_df, non_error_hap_df)

    #Record whether NN assignment was performed
    comb_stats_df.loc[comb_stats_df.sample_id.isin(test_samples), 'NN_assignment'] = 'yes'
    comb_stats_df.loc[comb_stats_df.NN_assignment.isnull(), 'NN_assignment'] = 'no'

    #Generate assignment hard calls if the threshold is met
    for level in ['coarse', 'int', 'fine']:
        asgn_dict = dict(results_df[level].loc[(results_df[level]>=float(nn_asgn_threshold)).any(axis=1)].apply(\
            lambda row: results_df[level].columns[row>=float(nn_asgn_threshold)][0], axis=1))
        comb_stats_df[f'res_{level}'] = comb_stats_df.sample_id.map(asgn_dict)

    comb_stats_df = estimate_contamination(comb_stats_df, non_error_hap_df, true_multi_targets, \
                                           rc_med_threshold, ma_med_threshold, ma_hi_threshold)

    return comb_stats_df

def generate_summary(comb_stats_df, version_name):

    summary = []
    summary.append(f'Nearest Neighbour assignment using reference version {version_name}')
    summary.append(f'On run containing {comb_stats_df.sample_id.nunique()} samples')
    summary.append(f'{(comb_stats_df.contamination_risk=="high").sum()} samples have high contamination risk')
    summary.append(f'{(comb_stats_df.contamination_risk=="medium").sum()} samples have medium contamination risk')
    summary.append(f'{(comb_stats_df.contamination_risk=="low").sum()} samples have low contamination risk')
    summary.append(f'{(comb_stats_df.NN_assignment=="no").sum()} samples with < 10 targets lack NN assignment')
    summary.append(f'{(~comb_stats_df.res_coarse.isnull()).sum()} samples are assigned at coarse level')
    summary.append(f'to {comb_stats_df.res_coarse.nunique()} different species groups')
    summary.append(f'{(~comb_stats_df.res_int.isnull()).sum()} samples are assigned at intermediate level')
    summary.append(f'to {comb_stats_df.res_int.nunique()} different species groups')
    summary.append(f'{(~comb_stats_df.res_fine.isnull()).sum()} samples are assigned at fine level')
    summary.append(f'to {comb_stats_df.res_fine.nunique()} different species groups')
    summary.append(f'{comb_stats_df.loc[(comb_stats_df.NN_assignment=="yes") & (comb_stats_df.res_int.isnull()), "sample_id"].nunique()} \
    samples with sufficient coverage could not be assigned at intermediate level')
    summary.append(f'{comb_stats_df.loc[(comb_stats_df.NN_assignment=="yes") & (comb_stats_df.res_int.isnull()) & (comb_stats_df.contamination_risk != "low"), "sample_id"].nunique()} \
    of those have medium or high contamination risk')
    summary.append(f'{comb_stats_df.loc[(comb_stats_df.NN_assignment=="yes") & (comb_stats_df.res_coarse.isnull()), "sample_id"].nunique()} \
    samples with sufficient coverage could not be assigned at coarse level')
    summary.append(f'{comb_stats_df.loc[(comb_stats_df.NN_assignment=="yes") & (comb_stats_df.res_coarse.isnull()) & (comb_stats_df.contamination_risk != "low"), "sample_id"].nunique()} \
    of those have medium or high contamination risk')
    
    return '\n'.join(summary)

def plot_assignment_proportions(nn_level_result_df, level_label, colors, nn_asgn_threshold):
    
    logging.info(f'Generate {level_label} level plots')
    #Generate bar plots at given assignment level
    width = min(20, .5*nn_level_result_df.shape[0])
    fig, ax = plt.subplots(figsize=(width,7))
    nn_level_result_df.plot(kind='bar', stacked=True, width=1, ax=ax, color=colors)
    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_title(f"{level_label} level assignment")
    ax.hlines(float(nn_asgn_threshold), -.5, nn_level_result_df.shape[0]-.5, color='k', ls = ':', linewidth=1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+3/7*box.height, box.width, box.height*4/7])
    leg1 = ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, -.05), fontsize=8.7)
    ax.margins(y=0)
    return fig, ax


def nn(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP NN data import started')

    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)
    logging.info(f'starting NN assignment for {comb_stats_df.sample_id.nunique()} samples on current run')
    mosq_hap_df = prep_mosquito_haps(hap_df, args.hap_read_count_threshold, \
                                     args.hap_reads_fraction_threshold)

    ref_hap_df, allele_freqs, true_multi_targets, \
        colors, version_name = prep_reference_index(\
        args.reference_version, path_to_refversion=args.path_to_refversion)

    kmers = construct_unique_kmer_table(mosq_hap_df, int(args.kmer_length))
    ref_kmers = construct_unique_kmer_table(ref_hap_df, int(args.kmer_length))

    error_seqs = identify_error_seqs(mosq_hap_df, kmers, int(args.kmer_length), int(args.n_error_snps))
    non_error_hap_df = mosq_hap_df[~mosq_hap_df.seqid.isin(error_seqs)]
    non_error_hap_df = recompute_haplotype_coverage(non_error_hap_df)
    non_error_hap_df.to_csv(f'{args.outdir}/non_error_haplotypes.tsv', index=False, sep='\t')
    
    nndict = find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers)
    nn_df = pd.DataFrame.from_dict(nndict, orient='index', columns=['nn_id_array', 'nn_dist'])
    nn_df['nn_id'] = ['|'.join(map(str, l)) for l in nn_df.nn_id_array]
    nn_df[['nn_id', 'nn_dist']].to_csv(f'{args.outdir}/nn_dictionary.tsv', sep='\t')

    results_df, test_samples = perform_nn_assignment_samples(\
        non_error_hap_df, ref_hap_df, nndict, allele_freqs, args.allelism_normalisation)
    for level in ['coarse', 'int', 'fine']:
        results_df[level].to_csv(f"{args.outdir}/assignment_{level}.tsv", sep='\t')

    comb_stats_df = generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, \
        results_df, true_multi_targets, args.nn_assignment_threshold, \
            args.medium_contamination_read_count_threshold, \
            args.medium_contamination_multi_allelic_threshold, \
            args.high_contamination_multi_allelic_threshold)

    logging.info(f'writing assignment results to {args.outdir}')
    comb_stats_df.to_csv(f'{args.outdir}/nn_assignment.tsv', index=False, sep='\t')
    
    summary_text = generate_summary(comb_stats_df, version_name)
    logging.info(f'writing summary file to {args.outdir}')
    with open(f'{args.outdir}/summary.txt', 'w') as fn:
        fn.write(summary_text)

    if not bool(args.no_plotting):
        for level in ['coarse', 'int', 'fine']:
            fig, _ = plot_assignment_proportions(results_df[level], level, colors[level], args.nn_assignment_threshold)
            fig.savefig(f'{args.outdir}/{level}_assignment.png')

    logging.info('ANOSPP NN complete')

    
def main():
    
    parser = argparse.ArgumentParser("NN assignment for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-r', '--reference_version', help='Reference index version - currently a directory name.\
         Default: nnv1', default='nnv1')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='nn')
    parser.add_argument('--path_to_refversion', help='path to reference index version.\
         Default: ref_databases', default='ref_databases')
    parser.add_argument('--no_plotting', help='Do not generate plots. Default: False', \
                        default=False)
    parser.add_argument('--allelism_normalisation', help='Normalisation method over multiple alleles. Options: \
                         [n_alleles,reads_fraction]. Default: n_alleles', default='n_alleles')
    parser.add_argument('--hap_read_count_threshold', help='minimum number of reads for supported haplotypes. \
         Default: 10', default=10)
    parser.add_argument('--hap_reads_fraction_threshold', help='minimum fraction of reads for supported haplotypes. \
         Default: 0.1', default=0.1)
    parser.add_argument('--medium_contamination_read_count_threshold', help='samples with fewer than this number \
                        of reads get medium contamination risk. Default: 1000', default=1000)
    parser.add_argument('--medium_contamination_multi_allelic_threshold', help='samples with more than this number \
                        of multiallelic targets get medium contamination risk. Default: 0', default=0)
    parser.add_argument('--high_contamination_multi_allelic_threshold', help='samples with more than this number \
                        of multiallelic targets get high contamination risk. Default: 2', default=2)
    parser.add_argument('--nn_assignment_threshold', help='required fraction for calling assignment. \
                        Default: 0.8', default=0.8)
    parser.add_argument('--n_error_snps', help='Maximum number of snps for a multi-allelic sequence to be \
                        considered a sequencing or PCR error. Default: 2', default=2)
    parser.add_argument('-k', '--kmer_length', help='Length of k-mers to use. Note that NNoVAE has been developed \
                        and tested for k=8, so accuracy of results cannot be guaranteed with other values of k. \
                        Default: k=8', default=8)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    nn(args)

if __name__ == '__main__':
    main()