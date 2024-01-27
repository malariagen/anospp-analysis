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



def plot_lims_plate(df, target, plate, fname, annot=True, cmap='coolwarm', title=None, center=None):

    """
    Plot a heatmap of the total read count for a given target on a plate.

    Args:
    - df (pandas.DataFrame): A dataframe containing the read counts and positions on the plate.
    - target (str): The name of the target (e.g. 'P1', 'P2').
    - plate (str): The name of the plate. Default is None.
    - fname (str): The name of the file to save the plot to. Default is None.
    - annot (bool): Whether to annotate the heatmap cells with their values. Default is True.
    - cmap (str or colormap): The colormap to use. Default is 'coolwarm'.
    - title (str): The title of the plot. Default is None.
    - center (float): The value at which to center the colormap. Default is None.

    Returns:
    - None
    """

    logging.info(f"plotting a heatmap for each lims plate for {target}")

    # Extract the column name that corresponds to the given target.
    col = f'total_reads_{target}'
    
    # Create a pivot table that maps the read counts to their respective positions on the plate.
    pivot_df = df.pivot(index='lims_row', columns='lims_col', values=col)
    
    # Set up the plot.
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot the heatmap using Seaborn.
    sns.heatmap(pivot_df, annot=annot, cmap=cmap, ax=ax, center=center, fmt='.5g')
    
    # Set the title of the plot.
    if not title:
        ax.set_title(f"Total {target} reads for {plate}")
    else:
        ax.set_title(title)

    # Add grid lines to the plot.
    ax.hlines([i * 2 for i in range(9)], 0, 24, colors='k')
    ax.vlines([j * 2 for j in range(13)], 0, 16, colors='k')
    
    # Adjust the layout of the plot to avoid overlapping.
    plt.tight_layout()
    
    # Save the plot to file if a filename is provided.
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    # Close the plot to free up memory.
    plt.close(fig)


def plot_bar(df, reference_path, fname):

    """
    Plots stacked bar charts of plasmodium species counts, grouped by plasmodium status for each plate.

    Args:
    - df: pandas DataFrame containing the plasmodium species and status counts for each plate
    - fname (str): file name to save the plot

    Returns:
    - None
    """

    logging.info(f"generating bar plots for the three plasmodium statuses")

    # Drop rows with missing values in 'plasmodium_species' or 'plasmodium_status'
    data = df.dropna(subset=['plasmodium_species', 'plasmodium_status'])

    # Group data by plasmodium species, status, and plate ID and count the occurrences
    plasmodium_count = pd.DataFrame({'count': data.groupby(["plasmodium_species", "plasmodium_status"]).size()}).reset_index()

    # Load colors
    if not os.path.isfile(f'{reference_path}/species_colours.csv'):
        logging.warning('No colors defined for plotting.')
        cmap = {}
    else:
        colors = pd.read_csv(f'{reference_path}/species_colours.csv')
        cmap = dict(zip(colors['species'], colors['color']))

    # Assign grey color to data with more than one species
    for index, row in plasmodium_count.iterrows():
        if len(row['plasmodium_species'].split(',')) > 1:
            cmap[row['plasmodium_species']] = '#cfcfcf'

    # Set up the plot
    plt.figure(figsize=(8, 8))
    sns.set_context(rc={'patch.linewidth': 0.5})

    # Create a bar plot
    ax = sns.catplot(x="plasmodium_species", y="count",
                 col="plasmodium_status", hue="plasmodium_species",
                 data=plasmodium_count, kind="bar", legend=False,
                 palette=cmap)

    # Customize plot labels
    ax.set_xticklabels(rotation=40, ha="right", fontsize=9)
    ax.set_xlabels('Plasmodium species predictions', fontsize=12)
    ax.set_ylabels('Species counts', fontsize=12)
    plt.tight_layout()

    # Save the plot to file
    plt.savefig(fname, dpi=300, bbox_inches='tight')


def hard_filter_haplotypes(hap_df, hard_filters):

    """
    Processes haplotype data and filters it according to read counts.

    Args:
    - hap_df (pd.DataFrame): Dataframe containing haplotype data.
    - hard_filters (int): A tupule containing the filters for the P1 and P2 targest

    Returns:
    - pd.DataFrame: Filtered dataframe containing haplotype data with read counts meeting the specified thresholds.
    """

    #unpack the hard filters
    filter_p1, filter_p2 = map(int, hard_filters.split(','))

    logging.info(f'filtering the haplotype data, cutoffs: P1 - {filter_p1}, P2 - {filter_p2}')
    
    # pull out the haplotypes that meet the filter_value
    p1_filter = (hap_df["target"] == "P1") & (hap_df["reads"] >= int(filter_p1))
    p2_filter = (hap_df["target"] == "P2") & (hap_df["reads"] >= int(filter_p2))
    filtered_hap_df =  hap_df[p1_filter | p2_filter]

    #Filter out columns that have no recorded samples
    haps_merged_df = filtered_hap_df.loc[:, (filtered_hap_df != 0).any(axis=0)]
    col_removed = len(filtered_hap_df.columns) - len(haps_merged_df.columns)
    logging.info(f'{col_removed} columns were removed for having no recorded samples')

    return haps_merged_df


def create_hap_data(hap_df):

    """
    Create a dataframe with haplotype and reads/sample stats from a haplotype dataframe.

    Args:
    - hap_df (pandas.DataFrame): The haplotype dataframe.

    Returns:
    - pandas.DataFrame: A dataframe with haplotype and reads/sample stats.
    """

    logging.info(f"creating a dataframe with haplotype and reads/sample stats.")

    # Create pivot table data for dada2 haplotypes using the hap_df data
    hap_data = hap_df[['sample_id', 'consensus', 'reads']].copy()
    hap_data_pivot = hap_data.pivot_table(values='reads', index='sample_id', columns='consensus')
    hap_data_pivot.fillna(0, inplace=True)

    # Remove 'consensus' header
    hap_data_pivot.columns.name = None

    # Move 'sample_id' index to column
    hap_data_pivot = hap_data_pivot.reset_index()

    # Filter out columns that have no recorded samples
    hap_data_pivot_filt = hap_data_pivot.loc[:, (hap_data_pivot != 0).any(axis=0)]

    haps_check = len(hap_data_pivot.columns) - len(hap_data_pivot_filt.columns)
    logging.info(f'{haps_check} columns had no recorded haplotype counts')

    # Set 'sample_id' as index and convert the values to integer
    hap_data_pivot_filt = hap_data_pivot_filt.set_index('sample_id').astype(int)

    return hap_data_pivot_filt


def haplotype_summary(hap_df, target, workdir):

    """
    Generate a summary of haplotype data for a specific target.

    Args:
    - hap_df (DataFrame): The input haplotype DataFrame.
    - target (str): The haplotype target.
    - workdir (str): The output directory for the summary (work).

    Returns:
    - haplotype_df (DataFrame): The haplotype summary DataFrame.
    - new_cols (list): The list of new column names for the summary DataFrame.
    """

    logging.info(f"processing the haplotype data for {target} and generating a table summary.")

    # Filter the input data to the current target and prepare the DataFrame.
    hap_df_filt = hap_df[hap_df["target"] == target]
    haps_df = create_hap_data(hap_df_filt)
    hap_df_filt = hap_df_filt.set_index("sample_id")
    haps_df_merged = pd.merge(
        left=hap_df_filt,
        left_index=True,
        right=haps_df,
        right_index=True,
        how="inner",
    )

    assert hap_df_filt.shape[0] == haps_df_merged.shape[0], 'Check your data as some data may have been lost or dropped'
    
    # Filter out columns that have no recorded samples.
    haps_df_merged = haps_df_merged.loc[:, (haps_df_merged != 0).any(axis=0)]

    # Rename the column of the combined DataFrame in place (automated!).
    new_cols = [
        f"haps_{target}_{i}" for i in range(0, (len(haps_df_merged.columns)) - len(hap_df_filt.columns))
    ]
    haps_df_merged.rename(
        columns=dict(zip(haps_df_merged.columns[len(hap_df_filt.columns) :], new_cols)),
        inplace=True,
    )

    # Drop duplicates (for where there are two haplotypes for one sample).
    haps_df_merged["index_col"] = haps_df_merged.index
    haps_df_merged.drop_duplicates(subset=["index_col", f"haps_{target}_0"], inplace=True)
    haps_df_merged = haps_df_merged.drop(columns=["index_col"])

    # Calculate the total reads and sample count per haplotype and append to original DataFrame.
    total_reads = (
        haps_df_merged.iloc[: len(haps_df_merged)]
        .select_dtypes(include=np.number)
        .sum()
        .rename("Total")
    )
    sample_count = pd.Series(
        haps_df_merged.iloc[: len(haps_df_merged)].astype(bool).sum(axis=0).rename("Sample_count"),
        index=haps_df_merged.columns,
    )
    haplotype_df = pd.concat(
        [haps_df_merged, total_reads.to_frame().T, sample_count.to_frame().T]
    ).rename_axis("sample_id")

    # Write out the haplotype DataFrame to a file.
    haplotype_df.to_csv(f"{workdir}/hap_{target}_uniq.tsv", sep="\t")

    return haplotype_df, new_cols


def run_blast(hap_data, target, workdir, path_to_refversion, reference_version):

    """
    Runs blast on haplotype data for a given target and returns a filtered dataframe.
    
    Args:
    - hap_data (pd.DataFrame): A pandas DataFrame containing haplotype data.
    - target (str): A string representing the target.
    - workdir (str): A string representing the working directory.
    - path_to_refversion, (str): A path to the reference directory.
    - reference_version, (str): A string representing the reference version to be used.
    
    Returns:
    - pd.DataFrame: A filtered pandas DataFrame containing the blast results for the haplotype data.
    """
    
    logging.info(f'running blast for {target}')
    
    # Filter the hapdata to the current targe
    hap_data = hap_data[hap_data['target'] == target]
    df = hap_data[['sample_id', 'target', 'reads', 'total_reads', 'reads_fraction', 'consensus']].copy().set_index('sample_id')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if target == 'P1':
            combuids = {cons: f"X1-{i}" for tgt, group in df.groupby(['target']) for i, cons in enumerate(group['consensus'].unique())}

        elif target == 'P2':
            combuids = {cons: f"X2-{i}" for tgt, group in df.groupby(['target']) for i, cons in enumerate(group['consensus'].unique())}
    
    # Add the combIDx and blast_id columns to the dataframe
    df['combUIDx'] = df['consensus'].astype(str).replace(combuids)
    df['blast_id'] = df.index.astype(str) + "." + df['combUIDx'].astype(str)

    # Convert the dataframe to fasta and run blast
    with open(f"{workdir}/comb_{target}_hap.fasta", "w") as output:
        for index, row in df.iterrows():
            output.write(">"+ index + "." + str(row['combUIDx'])+ "\n")
            output.write(row['consensus'] + "\n")

    # Load the blast db from the reference path
    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/plasmomito_P1P2_DB_v1.0.ndb'), f'reference version {reference_version} at {reference_path} \
        does not contain required plasmomito_P1P2_DB_v1.0.ndb file'
    
    blastdb = f'{path_to_refversion}/{reference_version}/plasmomito_P1P2_DB_v1.0'

    # Run blast and capture the output
    cmd = (
    f"blastn -db {blastdb} "
    f"-query {workdir}/comb_{target}_hap.fasta "
    f"-out {workdir}/comb_{target}_hap.tsv "
    f"-outfmt '6 qseqid sseqid slen qstart qend length mismatch gapopen gaps sseq, pident evalue bitscore qcovs' "
    f"-word_size 5 -max_target_seqs 1 -evalue 0.01"
        )
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # Handle errors
    if process.returncode != 0:
        logging.error(f"An error occurred while running the blastn command: {cmd}")
        logging.error(f"Command error: {process.stderr}")
        sys.exit(1)
    
    # Merge the blast results with the hap data and add additional columns
    blast_df = pd.read_csv(f'{workdir}/comb_{target}_hap.tsv', sep='\t', names=['qseqid', 'sseqid', 'slen', 'qstart', 'qend', 'length', 'mismatch',
                                                                                'gapopen', 'gaps', 'pident', 'evalue', 'bitscore', 'qcovs'])

    df = pd.merge(df.reset_index(), blast_df, how='right', left_on='blast_id', right_on='qseqid')

    df['genus'] = df.sseqid.str.split('_').str.get(0)
    df['specie'] = df.sseqid.str.split('_').str.get(1)
    df[f'ref_id_{target}'] = df['genus'] + '_' + df['specie']
    df['combUID'] = df.sseqid.str.split(':').str.get(1)

    # Subset the dataframe to only the needed columns
    blast_df = df[[
        'sample_id','target', 'reads', 'total_reads', 'reads_fraction', 'consensus',
        f'ref_id_{target}', 'combUID', 'combUIDx', 'length', 'pident', 'qcovs']].copy()
    
    # This line deals with a bug arising from an insufficiently long P. malariae reference sequence
    blast_df['hap_id'] = df.apply(lambda x: x.combUID if x.pident == 100 and ((x.combUID.startswith('M') and x.qcovs>=96) or (x.qcovs == 100)) else x.combUIDx, axis=1)

    return blast_df


def filter_blast(blast_df, target, soft_filters, filter_falciparum=False):

    """
    Filter Blast DataFrame based on specified criteria.

    Args:
    - blast_df (pd.DataFrame): Input DataFrame containing Blast results.
    - target (str): Target identifier ('P1' or 'P2').
    - soft_filters (str): Comma-separated string of soft filters.
    - filter_falciparum (bool): Whether to apply additional filters for Plasmodium falciparum.

    Returns:
    - pd.DataFrame: Filtered Blast DataFrame.
    """

    logging.info('Filtering Blast output')

    if filter_falciparum:
        filter_F1, filter_F2 = map(int, soft_filters.split(','))

        if target == 'P1':
            # Filter for 'F1-0' combUID and reads greater than or equal to filter_F1
            df_filtered = blast_df[(blast_df['combUID'] == 'F1-0') & (blast_df['reads'] >= filter_F1)]
            # Exclude rows with 'F1-0' combUID
            df_remaining = blast_df[blast_df['combUID'] != 'F1-0']

        elif target == 'P2':
            # Filter for 'F2-0' combUID and reads greater than or equal to filter_F2
            df_filtered = blast_df[(blast_df['combUID'] == 'F2-0') & (blast_df['reads'] >= filter_F2)]
            # Exclude rows with 'F2-0' combUID
            df_remaining = blast_df[blast_df['combUID'] != 'F2-0']

        blast_filt_df = pd.concat([df_filtered, df_remaining])

    else:
        # If no additional filters are applied, return a copy of the original DataFrame
        blast_filt_df = blast_df.copy()

    return blast_filt_df


def haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_filt_df, workdir):

    """
    Calculate haplotype diversity for a given target and write the results to a file.

    Args:
    - haplotype_df (pd.DataFrame): DataFrame containing haplotype data
    - target (str): Target (P1 or P2)
    - new_cols (list): List of new haplotype columns to add to the DataFrame
    - hap_df (pd.DataFrame): DataFrame containing haplotype information
    - blast_filt_df (pd.DataFrame): Filtered blast output DataFrame
    - outdir (str): Directory to write output file

    Returns:
    - pd.DataFrame: DataFrame with haplotype diversity information
    """

    logging.info(f'determining the haplotype diversity for {target}')

    # Filter the input data to the current target
    hap_df_filt = hap_df[hap_df['target'] == target]

    # Create the haplotype sequence dataframe
    haps_df = create_hap_data(hap_df_filt)
    hap_seq_df = pd.DataFrame({'haplotypes' :new_cols, 'sequences': haps_df.columns})

    # Create a new haplotype dataframe with haplotyes, total reads and sample_count columns
    hap_df_filt.set_index('sample_id', inplace=True)
    haplotypes = haplotype_df.columns[len(hap_df_filt.columns): ]
    total_reads = haplotype_df.iloc[-2, len(hap_df_filt.columns): ]
    sample_count = haplotype_df.iloc[-1, len(hap_df_filt.columns): ]
    hap_reads_stats_df = pd.DataFrame({'haplotypes': haplotypes, 'Total reads': total_reads, 'Sample_count': sample_count})

    # Merge the sequence and haplotype info dataframes and filter to only the target
    merged_hap_df = pd.merge(hap_seq_df, hap_reads_stats_df, on='haplotypes')

    # Convert Total reads and Sample count columns to integers
    merged_hap_df[['Total reads', 'Sample_count']] = merged_hap_df[['Total reads', 'Sample_count']].apply(
        lambda x: pd.to_numeric(x, errors='coerce').astype('Int64'))

    # Add the combUIDs to the above dataframe
    merged_hap_df_blast = blast_filt_df[['consensus', 'hap_id', f'ref_id_{target}']].copy()
    merged_hap_df_blast.drop_duplicates(subset=['consensus', 'hap_id'], inplace=True)

    # Merge haplotype and combUID dataframes
    hap_div_df = pd.merge(
        left=merged_hap_df, left_on='sequences', right=merged_hap_df_blast, right_on='consensus', how='right')

    # Write output to file
    hap_div_df.to_csv(f'{workdir}/Plasmodium_haplotype_summary_for_{target}.tsv', sep='\t', index=False)

    return hap_div_df


def generate_haplotype_tree(target, hap_div_df, workdir, outdir, interactive_plotting=False):

    """
    Generates an alignment, tree files, and a bokeh alignment plot for the haplotypes of a given target.

    Args:
    - target: The target to process.
    - hap_div_df: The DataFrame containing the haplotype data.
    - workdir: The path to the directory where haps data will be provisionally saved.
    - outdir: The path to the directory where the files will be saved.

    Returns:
    - None
    """

    logging.info(f'Generating the alignment, tree files, and bokeh alignment plots for {target}')

    # Create a fasta file from the haplotypes and perform a multiple sequence alignment
    hap_file = f'{workdir}/haps_{target}.fasta'
    mafft_out_file = f'{os.path.splitext(hap_file)[0]}_mafft.fasta'
    fasttree_out_file = f'{os.path.splitext(hap_file)[0]}_mafft.tre'

    with open(hap_file, 'w') as output:
        for index, row in hap_div_df.iterrows():
            output.write(">" + str(row[f'ref_id_{target}']) + '_' + str(row['hap_id']) + '_' + str(row['haplotypes']) + "\n")
            output.write(row['sequences']+"\n")

    # Generate a multiple sequence alignment using the fasta reads and mafft
    cmd_mafft = f'mafft --quiet --auto {hap_file} > {mafft_out_file}'
    run(cmd_mafft, shell=True)

    # Build the tree using fasttree
    cmd_fasttree = f'FastTree -quiet -nt -gtr -gamma {mafft_out_file} > {fasttree_out_file}'
    run(cmd_fasttree, shell=True)

    # Draw the tree and save it as a PNG image
    tree = Phylo.read(fasttree_out_file, 'newick')
    tree.ladderize(reverse=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    Phylo.draw(tree, axes=ax)
    fig.savefig(f'{outdir}/haps_{target}_mafft.png')
    plt.close(fig)

    # View and save the Alignment using bokeh
    if interactive_plotting:

        from .iplot import view_alignment

        aln_fn = mafft_out_file
        aln = AlignIO.read(aln_fn, 'fasta')
        aln_view_fn = f'{workdir}/haps_{target}_mafft.html'
        
        view_alignment(aln, aln_view_fn, fontsize="9pt", plot_width=1200)      


def create_per_read_summary(blast_filt_df, target, outdir):

    """
    Generates a per-read summary for the given target.

    Args:
    - blast_df (pd.DataFrame): The blast dataframe containing the reads.
    - target (str): The target to generate the summary for.
    - outdir (str): The directory to output the summary file to.

    Returns:
    - pd.DataFrame: The summary dataframe.
    """

    logging.info(f'generating a per-read summary for {target}')

    # Create a dataframe with the relevant stats
    df_sum = blast_filt_df.groupby(['sample_id', 'total_reads']).agg(
        {f'ref_id_{target}':lambda x: list(x), 'hap_id':lambda x: list(x), 'pident':lambda x: list(x),
         'reads':lambda x: list(x) ,'consensus': 'count'})
    
    # Rename column headers
    df_sum.reset_index(inplace=True)
    df_sum.set_index('sample_id', inplace=True)
    df_sum.rename(columns={'total_reads':f'total_reads_{target}', f'target':f'target_{target}', 'hap_id':f'haplotype_ID_{target}',
                              f'consensus':f'hap_count_{target}', 'pident':f'pident_{target}', 'reads':f'reads_{target}'}, inplace=True)

    # Write the dataframe to file
    df_sum.to_csv(f'{outdir}/results_summary_for_{target}.tsv', sep='\t')

    return df_sum


def merge_and_export(samples_df, merged_df, workdir):
    """
    Merge summary outputs with metadata and export the resulting dataframe to a specified directory.

    Args:
    - samples_df (pd.DataFrame): DataFrame containing metadata for each sample.
    - merged_df (pd.DataFrame): DataFrame containing summary outputs for each sample.
    - workdir (str): Directory to save the merged and exported dataframe.

    Returns:
    - pd.DataFrame: Merged and exported results with additional columns for sample run and sample ID.
    """

    logging.info(f'Merging summary outputs with sample metadata and exporting the merged dataframe to the work directory.')

    # Merge the two dataframes and select only relevant columns
    df_merged = pd.merge(samples_df.set_index('sample_id'), merged_df, left_index=True, right_index=True, how='right')

    # Define columns to keep
    cols_to_keep = ['plate_id']

    if 'total_reads_P1' in df_merged.columns:
        cols_to_keep += ['total_reads_P1', 'ref_id_P1', 'haplotype_ID_P1', 'pident_P1',
                         'reads_P1', 'hap_count_P1']

    if 'total_reads_P2' in df_merged.columns:
        cols_to_keep += ['total_reads_P2', 'ref_id_P2', 'haplotype_ID_P2', 'pident_P2',
                         'reads_P2', 'hap_count_P2']

    if 'sample_supplier_name' in df_merged.columns:
        cols_to_keep.insert(0, 'sample_supplier_name')

    df_final = df_merged[cols_to_keep].copy()

    # Add columns for sample ID
    df_final.index.name = 'sample_id'

    # Export the merged dataframe to a TSV file
    file_name = f'{workdir}/combined_results_summary.tsv'
    df_final.to_csv(file_name, sep='\t')

    return df_final


def process_results(haps_merged_df, hard_filters, workdir, outdir):

    """
    Read combined results summary TSV file and compute various metrics.

    Args:
    - haps_merged_df (pd.DataFrame): Filtered DataFrame containing haplotype data with read counts meeting the specified thresholds.
    - hard_filters (str): A tuple containing the filters for the P1 and P2 targets.
    - workdir (str): Working directory.
    - outdir (str): Directory containing the combined results summary TSV file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing various metrics for the samples.
    """

    logging.info(f'Reading and processing the combined results summary TSV file and computing stats.')

    def uniques(xs):
        return list(sorted(set(xi for x in xs for xi in x)))

    logging.info(f'reading results summary file and computing several metrics')

    # Unpack the hard filters
    filter_p1, filter_p2 = map(int, hard_filters.split(','))

    # Read the combined results summary results
    df = pd.read_csv(f'{workdir}/combined_results_summary.tsv', sep='\t').set_index('sample_id')

    # Create columns for fixing the read IDs
    for col in haps_merged_df['target'].unique():
        df[f'reads_{col}_name'] = df[f'ref_id_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else '')
        df[f'reads_{col}_fixed'] = df[f'reads_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else [0])
        df[f'pident_{col}_fixed'] = df[f'pident_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else [0])

    for col in haps_merged_df['target'].unique():
        df[f"{col}_min"] = df[f"reads_{col}_fixed"].apply(lambda x: min(int(y) for y in x) if x != ["0"] else 0)
        df[f"{col}_max"] = df[f"reads_{col}_fixed"].apply(lambda x: max(int(y) for y in x) if x != ["0"] else 0)
        df[f"{col}_avg"] = df[f"reads_{col}_fixed"].apply(lambda x: sum(int(y) for y in x) / len(x) if x != ["0"] else 0)
        df[f"{col}_min_pident"] = df[f"pident_{col}_fixed"].apply(lambda x: min(float(y) for y in x) if x != ["0"] else 0)
        df[f'hap_ID_{col}'] = df[f'haplotype_ID_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else '')

    # Compute concordance and species
    reads_cols = [col for col in ['reads_P1_name', 'reads_P2_name'] if col in df.columns]
    df['concordance'] = df[reads_cols].apply(uniques, axis=1).map(list)

    # Spread out the plasmodium id
    df_all = pd.merge(df, pd.DataFrame(df['concordance'].values.tolist()).add_prefix('plasmodium_id_'), on=df.index)

    # Set the index as sample_id
    df_all = df_all.rename(columns={'key_0': 'sample_id'}).set_index('sample_id')

    # Create a final species column detailing what the species are and remove comments.
    df_all["plasmodium_species"] = df_all.filter(regex="^plasmodium_id_").apply(lambda x: ", ".join(sorted(filter(lambda y: pd.notnull(y) and y != "", x))), axis=1)
    df_all["plasmodium_species"] = df_all["plasmodium_species"].str.replace("'", "")

    # Count the number of species per sample
    df_all['species_count'] = df['concordance'].apply(len)

    # Create Plasmodium status categories
    df_all['plasmodium_status'] = 'inconclusive'
    if 'P1_min' in df_all.columns and 'P2_min' in df_all.columns:
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] >= int(filter_p2)) &
                   (df_all['reads_P1_name'].apply(uniques) == df_all['reads_P2_name'].apply(uniques)), 'plasmodium_status'] = 'consistent'
        df_all.loc[(df_all['P1_min'] == 0) & (df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'P2 only'
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] == 0), 'plasmodium_status'] = 'P1 only'

    elif 'P1_min' in df_all.columns:
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)), 'plasmodium_status'] = 'P1 only'
     
    elif 'P2_min' in df_all.columns:
        df_all.loc[(df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'P2 only'

    # Create column for the presence of conflicts between P1 and P2
    if all(col in df_all.columns for col in ['reads_P1_name', 'reads_P2_name']):
        df_all['P1_P2_consistency'] = np.where(
            (df_all['reads_P1_name'].fillna('').apply(lambda x: len(set(x))) == 1) &
            (df_all['reads_P2_name'].fillna('').apply(lambda x: len(set(x))) == 1),
            'YES', 'NO')
    else:
        df_all['P1_P2_consistency'] = 'NO'
    
    # Create column for new haplotypes found
    # Check if both P1_min_pident and P2_min_pident are present in the dataframe
    if all(col in df_all.columns for col in ['P1_min_pident', 'P2_min_pident']):
        df_all['new_haplotype'] = np.where(
            ((df_all['P1_min_pident'].fillna(-1) < 100) & (df_all['P1_min_pident'].fillna(0) != 0)) |
            ((df_all['P2_min_pident'].fillna(-1) < 100) & (df_all['P2_min_pident'].fillna(0) != 0)),
            'YES', 'NO')
        
    # Check if only P1_min_pident is present in the dataframe
    elif 'P1_min_pident' in df_all.columns:
        df_all['new_haplotype'] = np.where(
            (df_all['P1_min_pident'].fillna(-1) < 100) & (df_all['P1_min_pident'].fillna(0) != 0), 'YES', 'NO')
    
    # Check if only P1_min_pident is present in the dataframe
    elif 'P2_min_pident' in df_all.columns:
        df_all['new_haplotype'] = np.where(
            df_all['P2_min_pident'].fillna(-1) < 100 & (df_all['P2_min_pident'].fillna(0) != 0), 'YES', 'NO') 

    # If none of the two columns are present, set new_haplotype to NO for all rows
    else:
        df_all['new_haplotype'] = 'NO'

    # Remove unwanted characters from hap_ID_P1 and hap_ID_P2 columns
    for col in haps_merged_df['target'].unique():
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r'\[|\]|"', '', regex=True)
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r"'", "")

        df_all[f'pident_{col}'] = df_all[f'pident_{col}'].astype(str).str.replace(r'\[|\]|"', '', regex=True)
        df_all[f'pident_{col}'] = (df_all[f'pident_{col}'].astype(str).str.replace(r"'", ""))

        df_all[f'reads_{col}'] = df_all[f'reads_{col}'].astype(str).str.replace(r'\[|\]|"', '', regex=True)
        df_all[f'reads_{col}'] = (df_all[f'reads_{col}'].astype(str).str.replace(r"'", ""))

    # Filter useful columns and save the results
    cols_to_keep = ['plate_id', 'plasmodium_species', 'plasmodium_status', 'species_count']

    if 'hap_ID_P1' in df_all.columns and 'pident_P1' in df_all.columns and 'reads_P1' in df_all.columns:
        cols_to_keep += ['hap_count_P1', 'total_reads_P1', 'hap_ID_P1', 'pident_P1', 'reads_P1']

    if 'hap_ID_P2' in df_all.columns and 'pident_P2' in df_all.columns and 'reads_P2' in df_all.columns:
        cols_to_keep += ['hap_count_P2', 'total_reads_P2', 'hap_ID_P2', 'pident_P2', 'reads_P2']

    if 'sample_supplier_name' in df_all.columns:
        cols_to_keep.insert(0, 'sample_supplier_name')

    df_all_final = df_all[cols_to_keep]

    # Replace 'nan' with '' in 'pident_' and 'reads_' columns if they exist
    for col in haps_merged_df['target'].unique():
        if f'pident_{col}' in df_all_final.columns:
            df_all_final.loc[:, f'pident_{col}'] = df_all_final[f'pident_{col}'].fillna('')

        if f'reads_{col}' in df_all_final.columns:
            df_all_final.loc[:, f'reads_{col}'] = df_all_final[f'reads_{col}'].fillna('')

    df_all_final.to_csv(f'{outdir}/plasmodium_predictions.tsv', sep='\t')

    return df_all_final


def generate_plots(meta_df_all, haps_merged_df, outdir, path_to_refversion, reference_version, interactive_plotting=False):
    
    """
    Generates plate and bar plots for the given metadata and haplotypes dataframes, and saves them in the specified
    directory.

    Args:
    - meta_df_all (pandas.DataFrame): A dataframe containing the metadata results.
    - haps_merged_df (pandas.DataFrame): A dataframe containing the haplotypes data.
    - outdir (str): The path of the directory to save the plots in.
    - path_to_refversion, (str): A path to the reference directory.
    - reference_version, (str): A string representing the reference version to be used.
    - interactive_plotting (bool): Whether to create interactive plots. Default is False.

    Returns:
    - None
    """

    logging.info('Generating plate and bar plots.')

    # Create columns for sorting the dataframe
    meta_df_all['lims_row'] = meta_df_all.lims_well_id.str.slice(0,1)
    meta_df_all['lims_col'] = meta_df_all.lims_well_id.str.slice(1).astype(int)


    # Get the lims plate IDs
    limsplate = meta_df_all.lims_plate_id.unique()

    # Create the reference path
    reference_path = f'{path_to_refversion}/{reference_version}/'

    # Check that the species_colours.csv file is present
    assert os.path.isfile(f'{reference_path}/species_colours.csv'), f'reference version {reference_version} at {reference_path} \
        does not contain required species_colours.csv file'

    # Make categorical plots for each lims plate
    for lims_plate in limsplate:
        for target in haps_merged_df['target'].unique():
            if interactive_plotting:
                plot_plate_view(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(), 
                                f'{outdir}/plateview_for_{lims_plate}_{target}.html',
                                target, reference_path,
                                f'{lims_plate} Plasmodium positive samples for {target}')


        #Make numerical plots for each lims plate
        for target in haps_merged_df['target'].unique():
            plot_lims_plate(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(),
                            target, lims_plate,
                            f'{outdir}/plateview_heatmap_{lims_plate}_{target}.png', annot=True)


    # Make the bar plots
    plot_bar(meta_df_all, reference_path, f'{outdir}/bar_plots.png')


def generate_stats(samples_df, haps_merged_df, merged_hap_df, df_all, outdir):

    """
    Generate summary statistics and write them to a file.

    Args:
    - samples_df (DataFrame): DataFrame containing sample data.
    - haps_merged_df (DataFrame): DataFrame containing merged haplotype data.
    - merged_hap_df (DataFrame): DataFrame containing merged haplotype data.
    - df_all (DataFrame): DataFrame containing all data.
    - outdir (str): Output directory path.

    Returns:
    - None
    """

    logging.info("Generating stats for this run")

    # Open the file for writing summary statistics
    stats_file_path = os.path.join(outdir, 'summary_stats.txt')
    
    with open(stats_file_path, 'w') as file:

        # Write the total number of samples
        file.write(f'Total sample count: {samples_df.index.nunique()}\n')

        # Write the number of samples positive for P1
        samples_positive_P1 = haps_merged_df[(haps_merged_df["target"] == "P1") & ~(haps_merged_df["target"] == "P2")]
        num_samples_positive_P1 = samples_positive_P1["sample_id"].nunique()
        file.write(f'Samples exclusively positive for P1: {num_samples_positive_P1}\n')

        # Write the number of samples positive for P2
        samples_positive_P2 = haps_merged_df[(haps_merged_df["target"] == "P2") & ~(haps_merged_df["target"] == "P1")]
        num_samples_positive_P2 = samples_positive_P2["sample_id"].nunique()
        file.write(f'Samples exclusively positive for P2: {num_samples_positive_P2}\n')

        # Write the number of samples positive for P1 or P2
        file.write(f'Sample positive for P1 or P2: {df_all.index.nunique()}\n')

        # Write the number of samples positive for both P1 and P2
        num_samples_positive_both = len(df_all[~df_all.isnull().any(axis=1)])
        file.write(f'Sample positive for P1 and P2: {num_samples_positive_both}\n')

        # Write the number of unique haplotypes for P1
        num_unique_hap_P1 = merged_hap_df[merged_hap_df["target"] == "P1"].consensus.nunique()
        file.write(f'Total count of unique P1 haplotypes: {num_unique_hap_P1}\n')

        # Write the number of unique haplotypes for P2
        num_unique_hap_P2 = merged_hap_df[merged_hap_df["target"] == "P2"].consensus.nunique()
        file.write(f'Total count of unique P2 haplotypes: {num_unique_hap_P2}\n')

        # Write the fraction of possible Plasmodium infection
        fraction_possible_infection = round(df_all.index.nunique() / samples_df.index.nunique() * 100, 2)
        file.write(f'Fraction of possible Plasmodium infections: {fraction_possible_infection} %\n')


def plasm(args):

    # Set up logging and create output directories
    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    # Prepare haplotype and sample dataframes
    logging.info('ANOSPP plasm data import started')
    hap_df = prep_hap(args.haplotypes)
    samples_df = prep_samples(args.manifest)

    # Combine haplotype and sample dataframes with stats
    logging.info('preparing input data and variables')
    haps_merged_df = hard_filter_haplotypes(hap_df, args.hard_filters)

    # Check for presence of PLASM_TARGETS
    logging.info('checking for the presence of PLASM_TARGETS')
    if len(haps_merged_df['target'].unique()) < 1:
        logging.warning('Could not find both PLASM_TARGETS in hap_df')
        sys.exit(1)

    try:
        # Run BLAST and create haplotype tree for each target
        logging.info('running blast')
        df_list = []
        hap_output = []
        for target in haps_merged_df['target'].unique():          
            haplotype_df, new_cols = haplotype_summary(hap_df, target, args.workdir)
            blast_df = run_blast(haps_merged_df, target, args.workdir, args.path_to_refversion, args.reference_version)
            blast_filt_df = filter_blast(blast_df, target, args.soft_filters, args.filter_falciparum)
            hap_div_df = haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_filt_df, args.workdir)
            generate_haplotype_tree(target, hap_div_df, args.workdir, args.outdir, args.interactive_plotting)
            per_read_summary = create_per_read_summary(blast_filt_df, target, args.workdir)
            df_list.append(per_read_summary)
            hap_output.append(blast_filt_df)

        # Combine blast results and create merged dataframe for all targets
        logging.info('merging blast summary outputs')
        merged_df = pd.concat(df_list, axis=1)
        merge_and_export(samples_df, merged_df, args.workdir)
        df_all = process_results(haps_merged_df, args.hard_filters, args.workdir, args.outdir)
        merged_hap_df = pd.concat(hap_output, axis=0)[['sample_id', 'hap_id', 'target', 'consensus', 'reads', 'pident']].copy().set_index('sample_id')
        merged_hap_df.to_csv(f'{args.outdir}/plasmodium_haplotypes.tsv', sep='\t')

        # generate the summary stats text file
        generate_stats(samples_df, haps_merged_df, merged_hap_df, df_all, args.outdir)

        # Merge sample and stats dataframes and generate plots
        logging.info('merging the samples(meta) dataframe with the stats dataframe and creating plots')
        meta_df_all = pd.merge(samples_df.set_index('sample_id'), df_all, left_index=True, right_index=True, how='left')
        generate_plots(meta_df_all, haps_merged_df, args.outdir, args.path_to_refversion, args.reference_version, args.interactive_plotting)

    except Exception as e:
        error_message = f'An error occurred: {e}'
        logging.error(error_message)
        raise RuntimeError(error_message)
    
    logging.info('ANOSPP plasm complete')


def main():


    parser = argparse.ArgumentParser("Plasmodium ID assignment for ANOSPP data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-p', '--path_to_refversion', help='path to reference index version.\
                        Default: ref_databases', default='ref_databases')
    parser.add_argument('-r', '--reference_version', help='Reference index version - currently a directory name. \
                        Default: plasmv1', default='plasmv1')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='plasm')
    parser.add_argument('-w', '--workdir', help='Working directory. Default: work', default='work')
    parser.add_argument('-f', '--hard_filters', help='Remove all sequences supported by less tahn X reads \
                        for P1 and P2. Default: 10,10', default='10,10')
    parser.add_argument('-g', '--soft_filters', help='Mark as non-confident any sequences of the predominant haplotype that are \
                        supported by fewer than X reads for P1 and P2. Default: 10,10', default='10,10')
    parser.add_argument('-i', '--interactive_plotting', 
                            help='do interactive plotting', action='store_true', default=False)
    parser.add_argument('--filter_falciparum', help='Check for the highest occuring haplotypes of Plasmodium falciparum and filter', 
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')


    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    plasm(args)


if __name__ == '__main__':
    main()

