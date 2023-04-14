import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import itertools

from .util import *

def prep_mosquito_haps(hap_df):
    '''
    prepare mosquito haplotype dataframe
    remove plasmodium haplotypes
    change targets to integers
    returns haplotype dataframe
    '''

    logging.info('preparing mosquito haplotypes')

    hap_df = hap_df.astype({'target': str})
    mosq_hap_df = hap_df[hap_df.target.isin(MOSQ_TARGETS)]
    mosq_hap_df = mosq_hap_df.astype({'target': int})

    return(mosq_hap_df)

def prep_reference_index(reference_dn, path_to_refversion='test_data'):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_dn}')

    reference_path = f'{path_to_refversion}/{reference_dn}/'

    assert os.path.isdir(reference_path), f'reference version {reference_dn} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/haplotypes.tsv'), f'reference version {reference_dn} at {reference_path} \
        does not contain required haplotypes.tsv file'
    ref_hap_df = pd.read_csv(f'{reference_path}haplotypes.tsv', sep='\t')

    assert os.path.isfile(f'{reference_path}/allele_freq_coarse.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_coarse.npy file'
    af_c = np.load(f'{reference_path}/allele_freq_coarse.npy')
    assert os.path.isfile(f'{reference_path}/allele_freq_int.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_int.npy file'
    af_i = np.load(f'{reference_path}/allele_freq_int.npy')
    assert os.path.isfile(f'{reference_path}/allele_freq_fine.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_fine.npy file'
    af_f = np.load(f'{reference_path}/allele_freq_fine.npy')

    assert os.path.isfile(f'{reference_path}/sgp_coarse.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_coarse.txt file'
    sgp_c = []
    with open(f'{reference_path}/sgp_coarse.txt', 'r') as fn:
        for line in fn:
            sgp_c.append(line.strip())

    assert os.path.isfile(f'{reference_path}/sgp_int.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_int.txt file'
    sgp_i = []
    with open(f'{reference_path}/sgp_int.txt', 'r') as fn:
        for line in fn:
            sgp_i.append(line.strip())

    assert os.path.isfile(f'{reference_path}/sgp_fine.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_fine.txt file'
    sgp_f = []
    with open(f'{reference_path}/sgp_fine.txt', 'r') as fn:
        for line in fn:
            sgp_f.append(line.strip())

    
    ref_hap_df['coarse_sgp'] = pd.Categorical(ref_hap_df['coarse_sgp'], sgp_c, ordered=True)
    ref_hap_df['intermediate_sgp'] = pd.Categorical(ref_hap_df['intermediate_sgp'], sgp_i, ordered=True)
    ref_hap_df['fine_sgp'] = pd.Categorical(ref_hap_df['fine_sgp'], sgp_f, ordered=True)

    assert os.path.isfile(f'{reference_path}/multiallelism.tsv'), f'reference version {reference_dn} at {reference_path} \
        does not contain required multiallelism.tsv file'
    true_multi_targets = pd.read_csv(f'{reference_path}/multiallelism.tsv', sep='\t')


    return(ref_hap_df, af_c, af_i, af_f, true_multi_targets)

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
    return(kmerdict)    

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

    maxallele = mosq_hap_df.groupby('target')['seqid'].nunique().max()
    kmerdict = construct_kmer_dict(k)
    
    uniqueseq = mosq_hap_df[['seqid', 'consensus']].drop_duplicates()

    table = np.zeros((len(MOSQ_TARGETS), maxallele, 4**k), dtype='int')
    for r in uniqueseq.index:
        seqid = str.split(uniqueseq.loc[r,'seqid'], '-')
        try:
            t, u = int(seqid[0]), int(seqid[1])
        except:
            raise Exception(f'seqid not recognised: {seqid}')
        sq = uniqueseq.loc[r,'consensus']
        for i in np.arange(len(sq)-(k-1)):
            table[t,u,kmerdict[sq[i:i+k]]] += 1
    return(table)

def parse_seqid(seqid_s):
    '''
    Parse seqids passed as a string or a pandas Series
    '''
    if isinstance(seqid_s, str):
        parsed_seqid = pd.DataFrame([seqid_s.split('-')])
    else:
        parsed_seqid = seqid_s.str.split('-', expand=True)
    
    assert parsed_seqid[0].isin(MOSQ_TARGETS).all(), 'Dataframe contains seqids referring to non-mosquito targets'
    try:
        parsed_seqid = parsed_seqid.astype(int)
    except:
        raise Exception('Dataframe contains seqids which cannot be converted to integers')
    return(parsed_seqid)

def identify_error_seqs(mosq_hap_df, kmers, k, error_snps = 2):
    '''
    Identify haplotypes resulting from sequencing/PCR errors
    Cannot distinguish between true heterozygote, contaminated homozygote and homozygote with error sequence
    So only look for errors for unique sequences at multiallelic targets
    '''

    logging.info('identifying haplotypes resulting from sequencing/PCR errors')
    #set the k-mer threshold for the number of snps allowed for errors
    threshold=error_snps*k+1
    seqid_size = mosq_hap_df.groupby('seqid').size()
    singleton_seqids = seqid_size[seqid_size==1].index
    error_candidates = mosq_hap_df.query('seqid in @singleton_seqids & nalleles>2')

    error_seqs = []
    for idx, cand in error_candidates.iterrows():
        possible_sources = mosq_hap_df.query('sample_id == @cand.sample_id & target == @cand.target & \
                                             not seqid in @error_seqs & seqid != @cand.seqid')
        cand_parsed_seqid = parse_seqid(cand.seqid)
        possible_sources_parsed_seqid = parse_seqid(possible_sources.seqid)
        for possible_source in possible_sources_parsed_seqid[1]:
            abs_kmer_dist = np.abs(kmers[cand.target,cand_parsed_seqid[1],:] - kmers[cand.target,possible_source,:]).sum()
            if abs_kmer_dist<threshold:
                error_seqs.append(cand.seqid)
                break
    
    logging.info(f'identified {len(error_seqs)} error sequences')

    return(error_seqs)
                        
def find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers):
    '''
    identify the nearest neighbours of the unique haplotypes in the reference dataset 
    '''
    maxima = ref_hap_df.groupby('target')['seqid'].nunique()

    nndict = dict()

    logging.info(f"identifying nearest neighbours for {non_error_hap_df.seqid.nunique()} unique haplotypes")
    
    for seqid in non_error_hap_df.seqid.unique():
        t, c = int(seqid.split('-')[0]), int(seqid.split('-')[1])
        #Compute difference between target and references, with length correction
        a = np.sum(np.abs(ref_kmers[t,:int(maxima[t]),:] - kmers[t,c,:]), axis=1)/ np.sum((
            ref_kmers[t,:int(maxima[t]),:] + kmers[t,c,:]), axis=1)
        #Find nearest neighbours
        cbn = np.arange(int(maxima[t]))[a==a.min()]
        #include in dict
        nndict[seqid] = (cbn, a.min())

    return(nndict)

def perform_nn_assignment_samples(non_error_hap_df, ref_hap_df, nndict, af_c, af_i, af_f, outdir):
    '''
    The main NN assignment function
    it outputs three dataframes containing the assignment proportions to each species-group for the three levels
    '''
    #get samples with at least 10 targets
    test_samples = non_error_hap_df.groupby('sample_id').filter(lambda x: x['target'].nunique() >=10)['sample_id'].unique()

    logging.info(f'performing NN assignment for {len(test_samples)} samples')

    #set up data-output as numpy arrays (will be made into dataframes later)
    res_coarse = np.zeros((len(MOSQ_TARGETS), len(test_samples), af_c.shape[2]))
    res_int = np.zeros((len(MOSQ_TARGETS), len(test_samples), af_i.shape[2]))
    res_fine = np.zeros((len(MOSQ_TARGETS), len(test_samples), af_f.shape[2]))

    for nsmp, smp in enumerate(test_samples):
        #Restrict to targets amplified in focal sample
        targets = non_error_hap_df.loc[non_error_hap_df.sample_id == smp, 'target'].unique()
        
        #Per amplified target
        for t in targets:
            #Identify the unique IDs of the focal sample's haplotypes at target t
            alleles = non_error_hap_df.loc[(non_error_hap_df.sample_id == smp) & (non_error_hap_df.target == t), 'seqid']
            #for each haplotype
            for allele in alleles:
                #lookup the nearest neighbour identifiers
                nnids = nndict[allele][0]
                #for each assignment level
                for table, lookup in zip([res_fine, res_int, res_coarse], [af_f, af_i, af_c]):
                    #get the (summed) allele frequences of the nearest neighbours
                    allele_freqs = lookup[t,nnids,:].sum(axis=0)
                    #normalise such that per sample neighbour frequency is 2 (diploid)
                    #and store in the results table
                    table[t,nsmp,:] += (2/len(alleles))*allele_freqs/np.sum(allele_freqs)
    
    #Make result tables into dataframes
    rc = np.nansum(res_coarse, axis=0)/np.sum(np.nansum(res_coarse, axis=0), axis=1)[:,None]
    result_coarse = pd.DataFrame(rc, index=test_samples, columns=ref_hap_df.coarse_sgp.cat.categories)
    result_coarse.to_csv(f"{outdir}/assignment_coarse.tsv", sep='\t')
    ri = np.nansum(res_int, axis=0)/np.sum(np.nansum(res_int, axis=0), axis=1)[:,None]
    result_intermediate = pd.DataFrame(ri, index=test_samples, columns=ref_hap_df.intermediate_sgp.cat.categories)
    result_intermediate.to_csv(f"{outdir}/assignment_intermediate.tsv", sep='\t')
    rf = np.nansum(res_fine, axis=0)/np.sum(np.nansum(res_fine, axis=0), axis=1)[:,None]
    result_fine = pd.DataFrame(rf, index=test_samples, columns=ref_hap_df.fine_sgp.cat.categories)
    result_fine.to_csv(f"{outdir}/assignment_fine.tsv", sep='\t')
        
    return(result_coarse, result_intermediate, result_fine, test_samples)

def recompute_coverage(comb_stats_df, non_error_hap_df):
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

    return(comb_stats_df)

def estimate_contamination(comb_stats_df, non_error_hap_df, true_multi_targets):
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

    comb_stats_df.loc[comb_stats_df.multiallelic_mosq_targets>2, 'contamination_risk'] = 'high'
    comb_stats_df.loc[((comb_stats_df.multiallelic_mosq_targets>0) & (comb_stats_df.multiallelic_mosq_targets<=2)) |\
        (comb_stats_df.mosq_reads<1000), 'contamination_risk'] = 'medium'
    comb_stats_df.loc[comb_stats_df.contamination_risk.isnull(), 'contamination_risk'] = 'low'

    logging.info(f"Identified {(comb_stats_df.contamination_risk=='high').sum()} samples with high contamination risk \
        \n and {(comb_stats_df.contamination_risk=='medium').sum()} samples with medium contamination risk")

    return(comb_stats_df)

def generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, result_coarse, \
                        result_int, result_fine, true_multi_targets):

    logging.info('generating NN calls from assignment info')

    comb_stats_df = recompute_coverage(comb_stats_df, non_error_hap_df)

    comb_stats_df.loc[comb_stats_df.sample_id.isin(test_samples), 'NN_assignment'] = 'yes'
    comb_stats_df.loc[comb_stats_df.NN_assignment.isnull(), 'NN_assignment'] = 'no'
    for result, rescol in zip([result_coarse, result_int, result_fine], ['res_coarse', 'res_int', 'res_fine']):
        adict = dict(result.loc[(result>=.8).any(axis=1)].apply(lambda row: result.columns[row>=0.8][0], axis=1))
        comb_stats_df[rescol] = comb_stats_df.sample_id.map(adict)

    comb_stats_df = estimate_contamination(comb_stats_df, non_error_hap_df, true_multi_targets)

    return(comb_stats_df)



def nn(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP NN data import started')

    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)

    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)
    mosq_hap_df = prep_mosquito_haps(hap_df)

    ref_hap_df, af_c, af_i, af_f, true_multi_targets = prep_reference_index(\
        args.reference, path_to_refversion=args.path_to_refversion)

    kmers = construct_unique_kmer_table(mosq_hap_df, k=8)
    ref_kmers = construct_unique_kmer_table(ref_hap_df, k=8)

    error_seqs = identify_error_seqs(mosq_hap_df, kmers, k=8)
    non_error_hap_df = mosq_hap_df[~mosq_hap_df.seqid.isin(error_seqs)]
    non_error_hap_df.to_csv(f'{args.outdir}/non_error_haplotypes.tsv', index=False, sep='\t')
    
    nndict = find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers)
    nn_df = pd.DataFrame.from_dict(nndict, orient='index', columns=['nn_id_array', 'nn_dist'])
    nn_df['nn_id'] = ['|'.join(map(str, l)) for l in nn_df.nn_id_array]
    nn_df[['nn_id', 'nn_dist']].to_csv(f'{args.outdir}/nn_dictionary.tsv', sep='\t')

    result_coarse, result_int, result_fine, test_samples = perform_nn_assignment_samples(non_error_hap_df, ref_hap_df, nndict, \
        af_c, af_i, af_f, args.outdir)

    comb_stats_df = generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, \
        result_coarse, result_int, result_fine, true_multi_targets)

    logging.info(f'writing assignment results to {args.outdir}')
    comb_stats_df.to_csv(f'{args.outdir}/nn_assignment.tsv', index=False, sep='\t')
    logging.info('All done!')

    

def main():
    
    parser = argparse.ArgumentParser("NN assignment for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-r', '--reference', help='Reference index version - currently a directory name.\
         Default: nn1.0', default='nn1.0')
    parser.add_argument('-p', '--path_to_refversion', help='path to reference index version.\
         Default: test_data', default='test_data')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='nn')
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    nn(args)

if __name__ == '__main__':
    main()