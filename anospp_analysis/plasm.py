import argparse
import os
import re
import sys
import subprocess

from anospp_analysis.util import *
from anospp_analysis.iplot import plot_plate_view

SUM_HAP_COLS = [
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
    'hap_seqid',
    'plate_id',
    'well_id',
    'consensus'
]

def run_blast(plasm_hap_df, outdir, blastdb, min_pident, min_qcov):

    logging.info('running blast')

    seq_df = plasm_hap_df[['seqid', 'consensus']].drop_duplicates()

    with open(f"{outdir}/plasm_haps.fasta", "w") as output:
        for _, row in seq_df.iterrows():
            output.write(f">{row['seqid']}\n")
            output.write(f"{row['consensus']}\n")

    # Run blast and capture the output
    blast_cols = 'qseqid sseqid slen qstart qend length mismatch gapopen gaps sseq pident evalue bitscore qcovs'
    cmd = (
        f"blastn -db {blastdb} "
        f"-query {outdir}/plasm_haps.fasta "
        f"-out {outdir}/plasm_blastout.tsv "
        f"-outfmt '6 {blast_cols}' "
        f"-word_size 5 -max_target_seqs 1 -evalue 0.01"
        )
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # Handle errors
    if process.returncode != 0:
        logging.error(f"An error occurred while running the blastn command: {cmd}")
        logging.error(f"Command error: {process.stderr}")
        sys.exit(1)

    blast_df = pd.read_csv(f'{outdir}/plasm_blastout.tsv', sep='\t', names=blast_cols.split())

    # not handling multiple blast hits for now
    multi_hits = blast_df.qseqid.duplicated()
    if multi_hits.any():
        multi_hits_haps = blast_df.qseqid[multi_hits].to_list()
        logging.warning(f'multiple blast hits found for {multi_hits_haps}, retaining only first hit for analysis')
        blast_df = blast_df[~ multi_hits]

    # annotate blast results
    blast_df['genus'] = blast_df.sseqid.str.split('_').str.get(0)
    blast_df['species'] = blast_df.sseqid.str.split('_').str.get(1)
    blast_df['binomial'] = blast_df['genus'] + '_' + blast_df['species']
    blast_df['species_assignment'] = blast_df['binomial']
    unknown_species = ((blast_df.pident < min_pident) | (blast_df.qcovs < min_qcov))
    blast_df.loc[unknown_species, 'species_assignment'] = 'unknown'
    blast_df['ref_seqid'] = blast_df.sseqid.str.split(':').str.get(1)

    def assign_hap_id(blast_row):
        
        # most annotations require both 100% coverage and identity    
        if blast_row.pident == 100:
            if blast_row.qcovs == 100:
                return blast_row.ref_seqid
            # M annotations require lower query coverage
            elif blast_row.ref_seqid.startswith('M') and (blast_row.qcovs >= min_qcov):
                return blast_row.ref_seqid
        
        # use per-per run seqids P1-0 -> X1-0 etc
        # seqids won't be sequential 
        hap_id_x = blast_row.qseqid.replace('P', 'X')

        return hap_id_x
        
    blast_df['hap_seqid'] = blast_df.apply(assign_hap_id, axis=1)

    return blast_df

def estimate_contamination(hap_df, comb_stats_df, min_samples, min_source_reads, max_affected_reads):
    """
    Identify potential contamination from excessive haplotype sharing between
    high coverage sample (source) and many low coverage samples (affected).

    Contamination is more likely between samples sharing plates or wells
    """

    logging.info('estimating cross-contamination')

    hap_df = hap_df[['sample_id', 'seqid', 'reads']]
    ext_hap_df = pd.merge(hap_df, comb_stats_df, on='sample_id', how='left')

    assert ~ext_hap_df['well_id'].isna().any(), 'failed to get well IDs'
    assert ~ext_hap_df['plate_id'].isna().any(), 'failed to get plate IDs'

    ext_hap_df['contamination_status'] = ''
    ext_hap_df['contamination_confidence'] = ''

    for seqid, hapid_df in ext_hap_df.groupby('seqid'):

        # status - haplotype sharing
        if (hapid_df.reads > min_source_reads).any():
            if (hapid_df.reads < max_affected_reads).sum() > min_samples:
                # source and affected data
                src_df = hapid_df.loc[hapid_df.reads > min_source_reads]
                tgt_df = hapid_df.loc[hapid_df.reads < max_affected_reads]
                # sample & hap define positions in original df
                src_haps = (ext_hap_df.sample_id.isin(src_df['sample_id']) & (ext_hap_df.seqid == seqid))
                tgt_haps = (ext_hap_df.sample_id.isin(tgt_df['sample_id']) & (ext_hap_df.seqid == seqid))
                # set contamination statuses in original df
                ext_hap_df.loc[(ext_hap_df.seqid == seqid), 'contamination_status'] = 'unclear'
                ext_hap_df.loc[src_haps, 'contamination_status'] = 'source'
                ext_hap_df.loc[tgt_haps, 'contamination_status'] = 'affected'
                # confidence - low without plate/well match
                ext_hap_df.loc[tgt_haps, 'contamination_confidence'] = 'low'
                for _, src_row in src_df.iterrows():
                    # affected samples sharing plate or well with source
                    same_plate_tgt_samples = tgt_df.loc[tgt_df.plate_id == src_row.plate_id, 'sample_id']
                    same_well_tgt_samples = tgt_df.loc[tgt_df.well_id == src_row.well_id, 'sample_id']
                    hc_tgt_samples = pd.concat([same_plate_tgt_samples, same_well_tgt_samples])
                    if len(hc_tgt_samples) > 0:
                        # sample & hap define positions in original df
                        hc_tgt_haps = (ext_hap_df.sample_id.isin(hc_tgt_samples) & (ext_hap_df.seqid == seqid))
                        # update confidence for plate/well match
                        ext_hap_df.loc[hc_tgt_haps, 'contamination_confidence'] = 'high'

    return ext_hap_df

def summarise_haplotypes(hap_df, blast_df, contam_df):

    logging.info('summarising haplotype info')

    sum_hap_df = pd.merge(hap_df, contam_df, how='left') # multiple columns to be merged
    sum_hap_df = pd.merge(sum_hap_df, blast_df, left_on='seqid', right_on='qseqid')

    sum_hap_df = sum_hap_df[SUM_HAP_COLS]

    return sum_hap_df

def summarise_samples(sum_hap_df, comb_stats_df, filters=(10,10)):

    logging.info('summarising sample info')

    sum_samples_df = comb_stats_df[[
        'sample_id',
        'sample_name',
        'lims_plate_id',
        'lims_well_id',
        'plate_id',
        'well_id'
    ]].copy().set_index('sample_id')

    sum_hap_df['reads_str'] = sum_hap_df['reads'].astype(str)
    for i, t in enumerate(PLASM_TARGETS):
        t_hap_df = sum_hap_df[sum_hap_df.target == t]
        t_sum_hap_gbs = t_hap_df.groupby('sample_id')
        sum_samples_df[f'{t}_reads_total'] = t_sum_hap_gbs['reads'].sum()
        sum_samples_df[f'{t}_reads_total'] = sum_samples_df[f'{t}_reads_total'].astype(float).fillna(0).astype(int)
        # pass criteria:
        # - read count over filter value
        # - haplotype is not high confidence affected by contamination
        t_pass_hap_gbs = t_hap_df[
            (t_hap_df.reads >= filters[i]) &
            (t_hap_df.contamination_confidence != 'high')
            ].sort_values('reads', ascending=False).groupby('sample_id')
        sum_samples_df[f'{t}_reads_pass'] = t_pass_hap_gbs['reads'].sum()
        sum_samples_df[f'{t}_reads_pass'] = sum_samples_df[f'{t}_reads_pass'].astype(float).fillna(0).astype(int)
        sum_samples_df[f'{t}_hapids_pass'] = t_pass_hap_gbs.agg({'hap_seqid': ';'.join})
        sum_samples_df[f'{t}_hapids_pass'] = sum_samples_df[f'{t}_hapids_pass'].fillna('')
        sum_samples_df[f'{t}_hapids_pass_reads'] = t_pass_hap_gbs.agg({'reads_str': ';'.join})
        sum_samples_df[f'{t}_hapids_pass_reads'] = sum_samples_df[f'{t}_hapids_pass_reads'].fillna('')
        sum_samples_df[f'{t}_species_assignments_pass'] = t_pass_hap_gbs.agg(
            {'species_assignment': ';'.join}
            )
        sum_samples_df[f'{t}_species_assignments_pass'] = sum_samples_df[f'{t}_species_assignments_pass'].fillna('')
        # contaminated haplotypes with read count over filter value
        t_contam_hap_gbs = t_hap_df[
            (t_hap_df.reads >= filters[i]) &
            (t_hap_df.contamination_status == 'affected') &
            (t_hap_df.contamination_confidence == 'high')
            ].sort_values('reads', ascending=False).groupby('sample_id')
        sum_samples_df[f'{t}_hapids_contam'] = t_contam_hap_gbs.agg({'hap_seqid': ';'.join})
        sum_samples_df[f'{t}_hapids_contam'] = sum_samples_df[f'{t}_hapids_contam'].fillna('')
        sum_samples_df[f'{t}_hapids_contam_reads'] = t_contam_hap_gbs.agg({'reads_str': ';'.join})
        sum_samples_df[f'{t}_hapids_contam_reads'] = sum_samples_df[f'{t}_hapids_contam_reads'].fillna('')
        # low coverage haplotypes
        t_locov_hap_gbs = t_hap_df[
            (t_hap_df.reads < filters[i])
            ].sort_values('reads', ascending=False).groupby('sample_id')
        sum_samples_df[f'{t}_hapids_locov'] = t_locov_hap_gbs.agg({'hap_seqid': ';'.join})
        sum_samples_df[f'{t}_hapids_locov'] = sum_samples_df[f'{t}_hapids_locov'].fillna('')
        sum_samples_df[f'{t}_hapids_locov_reads'] = t_locov_hap_gbs.agg({'reads_str': ';'.join})
        sum_samples_df[f'{t}_hapids_locov_reads'] = sum_samples_df[f'{t}_hapids_locov_reads'].fillna('')
        sum_samples_df[f'{t}_species_assignments_locov'] = t_locov_hap_gbs.agg(
            {'species_assignment': ';'.join}
            )
        sum_samples_df[f'{t}_species_assignments_locov'] = sum_samples_df[f'{t}_species_assignments_locov'].fillna('')
        

    def infer_status(sum_samples_row, targets=PLASM_TARGETS):
        # not generalised
        p1_spp_pass = set(sum_samples_row['P1_species_assignments_pass'].split(';')) - set([''])
        p2_spp_pass = set(sum_samples_row['P2_species_assignments_pass'].split(';')) - set([''])
        p1_spp_locov = set(sum_samples_row['P1_species_assignments_locov'].split(';')) - set([''])
        p2_spp_locov = set(sum_samples_row['P2_species_assignments_locov'].split(';')) - set([''])
        is_contam = (
            (len(sum_samples_row['P1_hapids_contam']) > 0) |
            (len(sum_samples_row['P2_hapids_contam']) > 0)
        )
        if len(p1_spp_pass) > 0:
            if len(p2_spp_pass) > 0:
                if p1_spp_pass == p1_spp_pass:
                    status = 'species_consistent'
                elif p1_spp_pass - p1_spp_pass == set():
                    status = 'extra_species_in_P2'
                elif p1_spp_pass - p1_spp_pass == set():
                    status = 'extra_species_in_P1'
                else:
                    status = 'species_discordant'
            else:
                # species consistent even if P2 does not pass coverage filter
                if p1_spp_pass == p2_spp_locov:
                    status = 'species_consistent_P2_locov'
                else:
                    status = 'P1_only'
        elif len(p2_spp_pass) > 0:
            if p1_spp_locov == p2_spp_pass:
                status = 'species_consistent_P1_locov'
            status = 'P2_only'
        elif is_contam:
            status = 'contamination_only'
        else:
            status = 'not_detected'

        return status

    sum_samples_df['plasmodium_detection_status'] = sum_samples_df.apply(infer_status, axis=1)

    def consensus_species(sum_samples_row, targets=PLASM_TARGETS):

        # no consensus species for statuses not considered positive
        nocall_statuses = (
            'contamination_only', 
            'P1_only', 
            'P2_only', 
            'not_detected'
            )
        if sum_samples_row['plasmodium_detection_status'] in nocall_statuses:
            return ''

        spp = set()
        for t in targets:
            tsp = set(sum_samples_row[f'{t}_species_assignments_pass'].split(';'))
            # locov only included when pass of same species is present, skip
            # tsp = tsp.union(set(sum_samples_row[f'{t}_species_assignments_locov'].split(';')))
            tsp = tsp - set([''])
            spp = spp.union(tsp)

        return ';'.join(spp)

    sum_samples_df['plasmodium_species'] = sum_samples_df.apply(consensus_species, axis=1)

    return sum_samples_df

def plasm(args):

    # Set up logging and create output directories
    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok=True)

    logging.info('ANOSPP plasm data import started')
    hap_df = prep_hap(args.haplotypes)
    run_id, comb_stats_df = prep_comb_stats(args.stats)

    plasm_hap_df = hap_df[hap_df['target'].isin(PLASM_TARGETS)].copy()
    
    reference_path = args.reference_path.rstrip('/')

    reference_version = reference_path.split('/')[-1]

    assert re.match(r'^plasmv\d', reference_version), f'{reference_version} not recognised as plasm ref version'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not exist at {reference_path}'


    if plasm_hap_df.shape[0] > 0:
        blastdb = f'{reference_path}/{args.blast_db_prefix}'

        blast_df = run_blast(
            plasm_hap_df, 
            args.outdir, 
            blastdb,
            args.blast_min_pident,
            args.blast_min_qcov
            )

        contam_df = estimate_contamination(
            plasm_hap_df, comb_stats_df,
            min_samples=args.contam_min_samples_affected, 
            min_source_reads=args.contam_min_reads_source, 
            max_affected_reads=args.contam_max_reads_affected
            )

        sum_hap_df = summarise_haplotypes(hap_df, blast_df, contam_df)

    # no plasmodium sequences in run - empty hap df
    else: 
        sum_hap_df = pd.DataFrame(columns=SUM_HAP_COLS)

    sum_hap_df.to_csv(f'{args.outdir}/plasm_hap_summary.tsv', sep='\t', index=False)

    sum_samples_df = summarise_samples(sum_hap_df, comb_stats_df, filters=(args.filter_p1, args.filter_p2))

    sum_samples_df['plasm_ref'] = reference_version

    out_df = sum_samples_df.drop(columns=[
        'sample_name',
        'lims_plate_id',
        'lims_well_id',
        'plate_id',
        'well_id',
        'P1_reads_total',
        'P2_reads_total'
    ]).copy()

    out_df.columns = out_df.columns.str.lower()

    out_df.to_csv(f'{args.outdir}/plasm_assignment.tsv', sep='\t')

    if args.interactive_plotting:
        for lims_plate in sum_samples_df.lims_plate_id.unique():

            logging.info(f'plotting interactive lims plate view for {lims_plate}')
            
            out_fn = f'{args.outdir}/plasm_{lims_plate}.html'
            title = f'Plasmodium species composition run {run_id}, plate {lims_plate}'
            plot_df = sum_samples_df[sum_samples_df.lims_plate_id == lims_plate]
            plot_plate_view(plot_df, out_fn, reference_path, title)

    logging.info('ANOSPP plasm complete')

    return


def main():

    parser = argparse.ArgumentParser("Plasmodium ID assignment for ANOSPP data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file generated by prep', required=True)
    parser.add_argument('-s', '--stats', help='Comb stats tsv file generated by prep', required=True)
    parser.add_argument('-r', '--reference_path', 
                        help='Path to plasm reference directory, expected to end with e.g. plasmv1', 
                        required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='plasm')
    parser.add_argument('--blast_db_prefix', 
                        help='Blast database prefix in reference index', 
                        default='plasmomito_P1P2_DB_v1.0')
    parser.add_argument('--blast_min_pident', 
                        help=('Minimum blast percent identity for haplotype assignment to species. '
                              'Default: 99 - corresponds to 2 SNPs or indels per ~210bp target'),
                        default=99, type=float)
    parser.add_argument('--blast_min_qcov', 
                        help='Minimum blast alignment query coverage for haplotype assignment to species. Default: 96',  
                        default=96, type=int)
    parser.add_argument('--contam_min_reads_source', 
                        help='Minimum number of reads in a source sample for same plate/well contamination. Default: 10000',  
                        default=10000, type=int)
    parser.add_argument('--contam_max_reads_affected', 
                        help='Maximum number of reads in a sample affected by same plate/well contamination. Default: 10000',  
                        default=100, type=int)
    parser.add_argument('--contam_min_samples_affected', 
                        help='Minimum number of samples affected by plate/well contamination to be considered. Default: 4',  
                        default=4, type=int)
    parser.add_argument('-f', '--filter_p1',
                        help='Minimum read support for P1 haplotypes to be included in sample summary. Default: 10',
                        default=10, type=int)
    parser.add_argument('-g', '--filter_p2', 
                        help='Minimum read support for P2 haplotypes to be included in sample summary. Default: 10', 
                        default=10, type=int)
    parser.add_argument('-i', '--interactive_plotting', 
                        help='Create interactive plots of species composition across plates', 
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')


    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    plasm(args)


if __name__ == '__main__':
    main()

