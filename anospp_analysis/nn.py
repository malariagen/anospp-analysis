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

    assert os.path.isfile(f'{reference_path}haplotypes.tsv'), f'reference version {reference_dn} at {reference_path} \
        does not contain required haplotypes.tsv file'
    ref_hap_df = pd.read_csv(f'{reference_path}haplotypes.tsv', sep='\t')

    assert os.path.isfile(f'{reference_path}allele_freq_coarse.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_coarse.npy file'
    af_c = np.load(f'{reference_path}/allele_freq_coarse.npy')
    assert os.path.isfile(f'{reference_path}allele_freq_int.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_int.npy file'
    af_i = np.load(f'{reference_path}/allele_freq_int.npy')
    assert os.path.isfile(f'{reference_path}allele_freq_fine.npy'), f'reference version {reference_dn} at {reference_path} \
        does not contain required allele_freq_fine.npy file'
    af_f = np.load(f'{reference_path}/allele_freq_fine.npy')

    assert os.path.isfile(f'{reference_path}sgp_coarse.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_coarse.txt file'
    sgp_c = []
    with open(f'{reference_path}sgp_coarse.txt', 'r') as fn:
        for line in fn:
            sgp_c.append(line.strip())

    assert os.path.isfile(f'{reference_path}sgp_int.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_int.txt file'
    sgp_i = []
    with open(f'{reference_path}sgp_int.txt', 'r') as fn:
        for line in fn:
            sgp_i.append(line.strip())

    assert os.path.isfile(f'{reference_path}sgp_fine.txt'), f'reference version {reference_dn} at {reference_path} \
        does not contain required sgp_fine.txt file'
    sgp_f = []
    with open(f'{reference_path}sgp_fine.txt', 'r') as fn:
        for line in fn:
            sgp_f.append(line.strip())

    
    ref_hap_df['coarse_sgp'] = pd.Categorical(ref_hap_df['coarse_sgp'], sgp_c, ordered=True)
    ref_hap_df['intermediate_sgp'] = pd.Categorical(ref_hap_df['intermediate_sgp'], sgp_i, ordered=True)
    ref_hap_df['fine_sgp'] = pd.Categorical(ref_hap_df['fine_sgp'], sgp_f, ordered=True)


    return(ref_hap_df, af_c, af_i, af_f)

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

def identify_error_seqs(mosq_hap_df, kmers, k):
    '''
    Identify haplotypes resulting from sequencing/PCR errors
    Cannot distinguish between true heterozygote, contaminated homozygote and homozygote with error sequence
    So only look for errors for unique sequences at multiallelic targets
    '''

    logging.info('identifying haplotypes resulting from sequencing/PCR errors')
    threshold=2*k+1
    seqid_size = mosq_hap_df.groupby('seqid').size()
    singleton_seqids = seqid_size[seqid_size==1].index
    error_candidates = singleton_seqids[(singleton_seqids.isin(mosq_hap_df.loc[mosq_hap_df.nalleles>2, 'seqid']))]

    error_seqs = []
    for tgt in pd.DataFrame(pd.Series(error_candidates).str.split('-', expand=True))[0].unique():
        for cand_hap in error_candidates[error_candidates.str.startswith(f'{tgt}-')]:
            cand_sample = mosq_hap_df.loc[mosq_hap_df.seqid==cand_hap, 'sample_id'].values[0]
            cand_sample_haps = mosq_hap_df.loc[(mosq_hap_df.sample_id==cand_sample) & (mosq_hap_df.target==int(tgt))]
            if cand_sample_haps.shape[0]>1:
                for i in cand_sample_haps.index[cand_sample_haps.seqid.isin(error_candidates)]:
                    for j in cand_sample_haps.index[(cand_sample_haps.index!=i) & (~cand_sample_haps.seqid.isin(error_seqs))]:
                        dist = np.abs(kmers[int(tgt), int(cand_sample_haps.loc[i, 'seqid'].split('-')[1]), :] - \
                            kmers[int(tgt), int(cand_sample_haps.loc[j, 'seqid'].split('-')[1]), :]).sum()
                        if dist<threshold:
                            error_seqs.append(cand_sample_haps.loc[i, 'seqid'])
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
        nndict[seqid] = cbn

    return(nndict)

def perform_nn_assignment_samples(non_error_hap_df, ref_hap_df, test_samples, nndict, af_c, af_i, af_f, outdir):
    '''
    The main NN assignment function
    it outputs three dataframes containing the assignment proportions to each species-group for the three levels
    '''
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
                nnids = nndict[allele]
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
    result_coarse.to_csv(f"{outdir}/assignment_coarse.csv")
    ri = np.nansum(res_int, axis=0)/np.sum(np.nansum(res_int, axis=0), axis=1)[:,None]
    result_intermediate = pd.DataFrame(ri, index=test_samples, columns=ref_hap_df.intermediate_sgp.cat.categories)
    result_intermediate.to_csv(f"{outdir}/assignment_intermediate.csv")
    rf = np.nansum(res_fine, axis=0)/np.sum(np.nansum(res_fine, axis=0), axis=1)[:,None]
    result_fine = pd.DataFrame(rf, index=test_samples, columns=ref_hap_df.fine_sgp.cat.categories)
    result_fine.to_csv(f"{outdir}/assignment_fine.csv")
        
    return(result_coarse, result_intermediate, result_fine)

def estimate_contamination(comb_stats_df, non_error_hap_df):
    '''
    estimate contamination from read counts and multiallelic targets
    '''
    logging.info('esimating contamination risk')

    comb_stats_df['multiallelic_mosq_targets'] = (non_error_hap_df.groupby('sample_id')['target'].value_counts() > 2\
        ).groupby(level='sample_id').sum()
    comb_stats_df['multiallelic_mosq_targets'] = comb_stats_df['multiallelic_targets'].fillna(0)

    #Exceptions -- target 32 for funestus
    funestus_32 = non_error_hap_df[(non_error_hap_df.sample_id.isin(comb_stats_df.loc[comb_stats_df.res_fine=='Anopheles_funestus'])) & \
        (non_error_hap_df.target==32)].groupby('sample_id')['seqid'].nunique()
    comb_stats_df.loc[comb_stats_df.sample_id.isin(funestus_32[funestus_32>2]), 'multiallelic_mosq_targets'] -= 1

    comb_stats_df.loc[comb_stats_df.multiallelic_mosq_targets>2, 'contamination_risk'] = 'high'
    comb_stats_df.loc[((comb_stats_df.multiallelic_mosq_targets>0) & (comb_stats_df.multiallelic_mosq_targets<=2)) |\
        (comb_stats_df.mosq_reads<1000), 'contamination_risk'] = 'medium'
    comb_stats_df.loc[comb_stats_df.contamination_risk.isnull(), 'contamination_risk'] = 'low'

    logging.info(f"Identified {(comb_stats_df.contamination_risk=='high').sum()} samples with high contamination risk \
        \n and {(comb_stats_df.contamination_risk=='medium').sum()} samples with medium contamination risk")

    return(comb_stats_df)

def generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, result_coarse, result_int, result_fine):

    logging.info('generating NN calls from assignment info')

    comb_stats_df.loc[comb_stats_df.sample_id.isin(test_samples), 'NN_assignment'] = 'yes'
    comb_stats_df.loc[comb_stats_df.NN_assignment.isnull(), 'NN_assignment'] = 'no'
    for result, rescol in zip([result_coarse, result_int, result_fine], ['res_coarse', 'res_int', 'res_fine']):
        adict = dict(result.loc[(result>=.8).any(axis=1)].apply(lambda row: result.columns[row>=0.8][0], axis=1))
        comb_stats_df[rescol] = comb_stats_df.sample_id.map(adict)

    comb_stats_df = estimate_contamination(comb_stats_df, non_error_hap_df)

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
    test_samples = comb_stats_df.loc[comb_stats_df.mosq_targets_recovered >= 10, 'sample_id'].values

    ref_hap_df, af_c, af_i, af_f = prep_reference_index(args.reference, path_to_refversion=args.path_to_refversion)

    kmers = construct_unique_kmer_table(mosq_hap_df, k=8)
    ref_kmers = construct_unique_kmer_table(ref_hap_df, k=8)

    error_seqs = identify_error_seqs(mosq_hap_df, kmers, k=8)
    non_error_hap_df = mosq_hap_df[~mosq_hap_df.seqid.isin(error_seqs)]
    non_error_hap_df.to_csv(f'{args.outdir}/non_error_haplotypes.tsv', index=False, sep='\t')
    
    nndict = find_nn_unique_haps(non_error_hap_df, kmers, ref_hap_df, ref_kmers)

    result_coarse, result_int, result_fine = perform_nn_assignment_samples(non_error_hap_df, ref_hap_df, test_samples, nndict, \
        af_c, af_i, af_f, args.outdir)

    comb_stats_df = generate_hard_calls(comb_stats_df, non_error_hap_df, test_samples, \
        result_coarse, result_int, result_fine)

    logging.info(f'writing assignment results to {args.outdir}')
    comb_stats_df.to_csv(f'{args.outdir}/nn_assignment.tsv', index=False, sep='\t')

    

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