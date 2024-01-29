import argparse
import os
import sys
import subprocess
from subprocess import run
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio import Phylo
import warnings

from anospp_analysis.util import *
from anospp_analysis.iplot import plot_plate_view

# blast
BLASTDB_PREFIX = 'plasmomito_P1P2_DB_v1.0'
BLAST_COLS = 'qseqid sseqid slen qstart qend length mismatch gapopen gaps sseq pident evalue bitscore qcovs'
# TODO check value against ref database
SPECIES_ASSIGNMENT_PIDENT = 97

# contamination estimation
MIN_AFFECTED_SAMPLES = 4
MAX_READS_AFFECTED_SAMPLE = 100
MIN_READS_SOURCE_SAMPLE = 10000

def run_blast(plasm_hap_df, outdir, path_to_refversion, reference_version, min_pident=SPECIES_ASSIGNMENT_PIDENT):

    logging.info('running blast')

    seq_df = plasm_hap_df[['seqid','consensus']].drop_duplicates()

    with open(f"{outdir}/plasm_haps.fasta", "w") as output:
        for _, row in seq_df.iterrows():
            output.write(f">{row['seqid']}\n")
            output.write(f"{row['consensus']}\n")

    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/{BLASTDB_PREFIX}.ndb'), f'reference version {reference_version} at {reference_path} \
        does not contain required {BLASTDB_PREFIX}.ndb file'

    blastdb = f'{path_to_refversion}/{reference_version}/{BLASTDB_PREFIX}'

    # Run blast and capture the output
    cmd = (
    f"blastn -db {blastdb} "
    f"-query {outdir}/plasm_haps.fasta "
    f"-out {outdir}/plasm_blastout.tsv "
    f"-outfmt '6 {BLAST_COLS}' "
    f"-word_size 5 -max_target_seqs 1 -evalue 0.01"
        )
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # Handle errors
    if process.returncode != 0:
        logging.error(f"An error occurred while running the blastn command: {cmd}")
        logging.error(f"Command error: {process.stderr}")
        sys.exit(1)

    blast_df = pd.read_csv(f'{outdir}/plasm_blastout.tsv', sep='\t', names=BLAST_COLS.split())

    # not handling multiple blast hits for now
    assert blast_df.qseqid.is_unique, f'multiple blast hits found, see {outdir}/plasm_blastout.tsv'

    # annotate blast results
    blast_df['genus'] = blast_df.sseqid.str.split('_').str.get(0)
    blast_df['species'] = blast_df.sseqid.str.split('_').str.get(1)
    blast_df['binomial'] = blast_df['genus'] + '_' + blast_df['species']
    blast_df['species_assignment'] = blast_df['binomial']
    unknown_species = (blast_df['pident'] < min_pident)
    blast_df.loc[unknown_species, 'species_assignment'] = 'unknown'
    blast_df['ref_seqid'] = blast_df.sseqid.str.split(':').str.get(1)

    def assign_hap_id(blast_row):

        if blast_row.pident == 100:
            # most annotations require both 100% coverage and identity
            if blast_row.qcovs == 100:
                return blast_row.ref_seqid
            # M annotations require lower query coverage
            elif blast_row.ref_seqid.startswith('M') and blast_row.qcovs>=96:
                return blast_row.ref_seqid
        
        # use per-per run seqids P1-0 -> X1-0 etc
        # seqids won't be sequential 
        hap_id_x = blast_row.qseqid.replace('P','X')

        return hap_id_x
        
    blast_df['hap_seqid'] = blast_df.apply(assign_hap_id, axis=1)

    return blast_df

def estimate_contamination(hap_df, sample_df, 
                           min_samples=MIN_AFFECTED_SAMPLES, 
                           min_source_reads=MIN_READS_SOURCE_SAMPLE, 
                           max_target_reads=MAX_READS_AFFECTED_SAMPLE):
    """
    Identify potential contamination from excessive haplotype sharing between
    high coverage sample (source) and many low coverage samples (targets).

    Contamination is more likely between samples sharing plates or wells
    """

    logging.info('estimating cross-contamination')

    hap_df = hap_df[['sample_id', 'seqid', 'reads']]
    ext_df = sample_df[['sample_id', 'plate_id', 'well_id']]
    ext_hap_df = pd.merge(hap_df, sample_df, on='sample_id', how='left')

    assert ~ext_hap_df['well_id'].isna().any(), 'failed to get well IDs'
    assert ~ext_hap_df['plate_id'].isna().any(), 'failed to get plate IDs'

    ext_hap_df['contamination_status'] = ''
    ext_hap_df['contamination_confidence'] = ''

    for seqid, hapid_df in ext_hap_df.groupby('seqid'):

        # status - haplotype sharing
        if (hapid_df.reads > min_source_reads).any():
            if (hapid_df.reads < max_target_reads).sum() > min_samples:
                # source and target data
                src_df = hapid_df.loc[hapid_df.reads > min_source_reads]
                tgt_df = hapid_df.loc[hapid_df.reads < max_target_reads]
                # sample & hap define positions in original df
                src_haps = (ext_hap_df.sample_id.isin(src_df['sample_id']) & (ext_hap_df.seqid == seqid))
                tgt_haps = (ext_hap_df.sample_id.isin(tgt_df['sample_id']) & (ext_hap_df.seqid == seqid))
                # set contamination statuses in original df
                ext_hap_df.loc[(ext_hap_df.seqid == seqid), 'contamination_status'] = 'unclear'
                ext_hap_df.loc[src_haps, 'contamination_status'] = 'source'
                ext_hap_df.loc[tgt_haps, 'contamination_status'] = 'target'
                # confidence - plate/well match
                ext_hap_df.loc[tgt_haps, 'contamination_confidence'] = 'low'
                for _, src_row in src_df.iterrows():
                    # targets sharing plate or well with source
                    same_plate_tgt_samples = tgt_df.loc[tgt_df.plate_id == src_row.plate_id, 'sample_id']
                    same_well_tgt_samples = tgt_df.loc[tgt_df.well_id == src_row.well_id, 'sample_id']
                    hc_tgt_samples = pd.concat([same_plate_tgt_samples, same_well_tgt_samples])
                    if len(hc_tgt_samples) > 0:
                        # sample  & hap define positions in original df
                        hc_tgt_haps = (ext_hap_df.sample_id.isin(hc_tgt_samples) & (ext_hap_df.seqid == seqid))
                        # update confidence 
                        ext_hap_df.loc[hc_tgt_haps, 'contamination_confidence'] = 'high'

    return ext_hap_df

def summarise_haplotypes(hap_df, blast_df, contam_df):

    logging.info('summarising haplotype info')

    sum_hap_df = pd.merge(hap_df, contam_df, how='left') # multiple columns to be merged
    sum_hap_df = pd.merge(sum_hap_df, blast_df, left_on='seqid', right_on='qseqid')

    sum_hap_df = sum_hap_df[[
        'sample_id',
        'target',
        'reads',
        'total_reads',
        'reads_fraction',
        'nalleles',
        'seqid',
        'sample_name',
        'contamination_status',
        'contamination_confidence',
        'sseqid',
        'pident',
        'qcovs',
        'species_assignment',
        'hap_seqid'
    ]]

    return sum_hap_df

def summarise_samples(sum_hap_df, samples_df, filters=(10,10)):

    logging.info('summarising sample info')

    sum_samples_df = samples_df[[
        'sample_id',
        'sample_name',
        'lims_plate_id',
        'lims_well_id',
        'plate_id',
        'well_id'
    ]].copy().set_index('sample_id')

    for i, t in enumerate(PLASM_TARGETS):
        t_hap_df = sum_hap_df[sum_hap_df.target == t]
        t_sum_hap_gbs = t_hap_df.groupby('sample_id')
        sum_samples_df[f'{t}_reads_total'] = t_sum_hap_gbs['reads'].sum()
        sum_samples_df[f'{t}_reads_total'] = sum_samples_df[f'{t}_reads_total'].fillna(0).astype(int)
        # pass criteria:
        # - read count over filter value
        # - haplotype is not high confidence target of contamination
        t_pass_hap_gbs = t_hap_df[
            (t_hap_df.reads >= filters[i]) &
            (t_hap_df.contamination_status != 'target') &
            (t_hap_df.contamination_confidence != 'high')
            ].groupby('sample_id')
        sum_samples_df[f'{t}_reads_pass'] = t_pass_hap_gbs['reads'].sum()
        sum_samples_df[f'{t}_reads_pass'] = sum_samples_df[f'{t}_reads_pass'].fillna(0).astype(int)
        sum_samples_df[f'{t}_haps_pass'] = t_pass_hap_gbs['seqid'].nunique()
        sum_samples_df[f'{t}_haps_pass'] = sum_samples_df[f'{t}_haps_pass'].fillna(0).astype(int)
        sum_samples_df[f'{t}_hapids_pass'] = t_pass_hap_gbs.agg({'hap_seqid': ','.join})
        sum_samples_df[f'{t}_hapids_pass'] = sum_samples_df[f'{t}_hapids_pass'].fillna('')
        sum_samples_df[f'{t}_species_assignments_pass'] = t_pass_hap_gbs.agg(
            {'species_assignment': ','.join})
        sum_samples_df[f'{t}_species_assignments_pass'] = sum_samples_df[f'{t}_species_assignments_pass'].fillna('')
        # contaminated sequences with read count over filter value
        t_contam_hap_gbs = t_hap_df[
            (t_hap_df.reads >= filters[i]) &
            (t_hap_df.contamination_status == 'target') &
            (t_hap_df.contamination_confidence == 'high')
            ].groupby('sample_id')
        sum_samples_df[f'{t}_hapids_contam'] = t_contam_hap_gbs.agg({'hap_seqid': ','.join})
        sum_samples_df[f'{t}_hapids_contam'] = sum_samples_df[f'{t}_hapids_contam'].fillna('')

    def infer_status(sum_samples_row, targets=PLASM_TARGETS):
        # not generalised
        p1_spp = set(sum_samples_row['P1_species_assignments_pass'].split(',')) - set([''])
        p2_spp = set(sum_samples_row['P2_species_assignments_pass'].split(',')) - set([''])
        is_contam = (
            (len(sum_samples_row['P1_hapids_contam']) > 0) |
            (len(sum_samples_row['P2_hapids_contam']) > 0)
        )
        if len(p1_spp) > 0:
            if len(p2_spp) > 0:
                if p1_spp == p2_spp:
                    status = 'consistent'
                elif p1_spp - p2_spp == set():
                    status = 'extra_species_in_P1'
                elif p2_spp - p1_spp == set():
                    status = 'extra_species_in_P2'
                else:
                    status = 'discordant'
            else:
                status = 'P1_only'
        elif len(p2_spp) > 0:
            status = 'P2_only'
        elif is_contam:
            status = 'contamination'
        else:
            status = 'no_infection'

        return status

    sum_samples_df['plasmodium_status'] = sum_samples_df.apply(infer_status, axis=1)

    def consensus_species(sum_samples_row, targets=PLASM_TARGETS):

        spp = set()
        for t in targets:
            tsp = set(sum_samples_row[f'{t}_species_assignments_pass'].split(',')) - set([''])
            spp = spp.union(tsp)

        return ','.join(spp)

    sum_samples_df['plasmodium_species'] = sum_samples_df.apply(consensus_species, axis=1)

    return sum_samples_df

def plasm(args):

    # Set up logging and create output directories
    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok=True)
    # os.makedirs(args.workdir, exist_ok=True)

    filters = [int(i) for i in args.filters.split(',')]
    # TODO verify ref

    logging.info('ANOSPP plasm data import started')
    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)

    plasm_hap_df = hap_df[hap_df['target'].isin(PLASM_TARGETS)].copy()

    blast_df = run_blast(
        plasm_hap_df, args.outdir, 
        args.path_to_refversion, args.reference_version
        )

    contam_df = estimate_contamination(
        plasm_hap_df, samples_df,
        min_samples=MIN_AFFECTED_SAMPLES, 
        min_source_reads=MIN_READS_SOURCE_SAMPLE, 
        max_target_reads=MAX_READS_AFFECTED_SAMPLE)

    sum_hap_df = summarise_haplotypes(hap_df, blast_df, contam_df)

    sum_hap_df.to_csv(f'{args.outdir}/plasm_hap_summary.tsv', sep='\t', index=False)

    sum_samples_df = summarise_samples(sum_hap_df, samples_df, filters)

    sum_samples_df.to_csv(f'{args.outdir}/plasm_sample_summary.tsv', sep='\t')

    logging.info('ANOSPP plasm complete')

    return

def main():


    parser = argparse.ArgumentParser("Plasmodium ID assignment for ANOSPP data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-p', '--path_to_refversion', help='path to reference index version.\
                        Default: ref_databases', default='ref_databases')
    parser.add_argument('-r', '--reference_version', help='Reference index version - currently a directory name. \
                        Default: plasmv1', default='plasmv1')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='plasm')
    # parser.add_argument('-w', '--workdir', help='Working directory. Default: work', default='work')
    # parser.add_argument('-f', '--hard_filters', help='Remove all sequences supported by less tahn X reads \
    #                     for P1 and P2. Default: 10,10', default='10,10')
    parser.add_argument('-f', '--filters', help='Mark as non-confident any sequences of the predominant haplotype that are \
                        supported by fewer than X reads for P1 and P2. Default: 10,10', default='10,10')
    # parser.add_argument('-i', '--interactive_plotting', 
    #                         help='do interactive plotting', action='store_true', default=False)
    # parser.add_argument('--filter_falciparum', help='Check for the highest occuring haplotypes of Plasmodium falciparum and filter', 
    #                     action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')


    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    plasm(args)


if __name__ == '__main__':
    main()

