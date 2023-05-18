import argparse
import os
import subprocess
from subprocess import run
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio import Phylo

from .util import *
from .iplot import plot_plate_view


def plot_lims_plate(df, target, plate, fname, annot=True, cmap='coolwarm', title=None, center=None):
    """
    Plot a heatmap of the total read count for a given target on a plate.

    Args:
        df (pandas.DataFrame): A dataframe containing the read counts and positions on the plate.
        target (str): The name of the target (e.g. 'P1', 'P2').
        plate (str): The name of the plate. Default is None.
        fname: The name of the file to save the plot to. Default is None.
        annot (bool): Whether to annotate the heatmap cells with their values. Default is True.
        cmap (str or colormap): The colormap to use. Default is 'coolwarm'.
        center (float): The value at which to center the colormap. Default is None.
        title (str): The title of the plot. Default is None.

    Returns:
        None
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

def plot_bar(df, reference, fname):
    """
    Plots stacked bar charts of plasmodium species counts, grouped by plasmodium status for each plate.

    Args:
    - df: pandas DataFrame containing the plasmodium species and status counts for each plate
    - fname: str, file name to save the plot

    Returns:
    - None
    """

    logging.info(f"generating bar plots for the three plasmodium statuses")

    # Drop rows with missing values in 'plasmodium_species' or 'plasmodium_status'
    data = df.dropna(subset=['plasmodium_species', 'plasmodium_status'])

    # Group data by plasmodium species, status, and plate ID and count the occurrences
    plasmodium_count = pd.DataFrame({'count': data.groupby(["plasmodium_species", "plasmodium_status"]).size()}).reset_index()

    # Load colors
    if not os.path.isfile(f'{reference}/species_colours.csv'):
        logging.warning('No colors defined for plotting.')
        cmap = {}
    else:
        colors = pd.read_csv(f'{reference}/species_colours.csv')
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
                     col="plasmodium_status", data=plasmodium_count,
                     kind="bar", facet_kws={'legend_out': True},
                     palette=cmap)

    # Customize plot labels
    ax.set_xticklabels(rotation=40, ha="right", fontsize=9)
    ax.set_xlabels('Plasmodium species predictions', fontsize=12)
    ax.set_ylabels('Species counts', fontsize=12)
    plt.tight_layout()

    # Save the plot to file
    plt.savefig(fname, dpi=300, bbox_inches='tight')


def hard_filter_haplotypes(hap_df, filter_p1, filter_p2):

    """
    Processes haplotype data and filters it according to read counts.

    Args:
        hap_df (pd.DataFrame): Dataframe containing haplotype data.
        comb_stats_df (pd.DataFrame): Dataframe containing various useful statistics.
        filter_p1 (int): Minimum read count for haplotypes associated with target P1.
        filter_p2 (int): Minimum read count for haplotypes associated with target P2.

    Returns:
        pd.DataFrame: Filtered dataframe containing haplotype data with read counts meeting the specified thresholds.
    """

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
        hap_df (pandas.DataFrame): The haplotype dataframe.

    Returns:
        pandas.DataFrame: A dataframe with haplotype and reads/sample stats.
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
    # assert haps_df.loc[:,(haps_df == 0).all(axis=0)], f"columns with no columns are not expected here. Check your input data" ### get Alex's assistance

    haps_check = len(hap_data_pivot.columns) - len(hap_data_pivot_filt.columns)
    logging.info(f'{haps_check} columns had no recorded haplotype counts')

    # Set 'sample_id' as index and convert the values to integer
    hap_data_pivot_filt = hap_data_pivot_filt.set_index('sample_id').astype(int)

    return hap_data_pivot_filt


def haplotype_summary(hap_df, target, workdir):
    """
    Generate a summary of haplotype data for a specific target.

    Parameters:
    -----------
    hap_df : pandas DataFrame
        The input haplotype DataFrame.
    target : str
        The haplotype target.
    workdir : str
        The output directory for the summary (work).

    Returns:
    --------
    haplotype_df : pandas DataFrame
        The haplotype summary DataFrame.
    new_cols : list
        The list of new column names for the summary DataFrame.
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

def run_blast(hap_data, target, workdir, blastdb):

    """
    Runs blast on haplotype data for a given target and returns a filtered dataframe.
    
    Args:
    hap_data (pd.DataFrame): A pandas DataFrame containing haplotype data.
    target (str): A string representing the target.
    workdir (str): A string representing the working directory.
    blastdb (str): A string representing the path to the blast database.
    filter_F1 (int): An integer representing the filter for target P1. Default is 10.
    filter_F2 (int): An integer representing the filter for target P2. Default is 10.
    
    Returns:
    pd.DataFrame: A filtered pandas DataFrame containing the blast results for the haplotype data.
    """

    logging.info(f'running blast for {target}')
    
    #filter the hapdata to the current targe
    hap_data = hap_data[hap_data['target'] == target]
    df = hap_data[['sample_id', 'target', 'reads', 'total_reads', 'reads_fraction', 'consensus']].copy().set_index('sample_id')
    if target == 'P1':
        combuids = {cons: f"X1-{i}" for tgt, group in df.groupby(['target']) for i, cons in enumerate(group['consensus'].unique())}
    elif target == 'P2':
        combuids = {cons: f"X2-{i}" for tgt, group in df.groupby(['target']) for i, cons in enumerate(group['consensus'].unique())}
    
    df['combUIDx'] = df['consensus'].astype(str).replace(combuids)
    df['blast_id'] = df.index.astype(str) + "." + df['combUIDx'].astype(str)


    #convert the dataframe to fasta and run blast
    with open(f"{workdir}/comb_{target}_hap.fasta", "w") as output:
        for index, row in df.iterrows():
            output.write(">"+ index + "." + str(row['combUIDx'])+ "\n")
            output.write(row['consensus'] + "\n")

    # Run blast and capture the output
    cmd = f"blastn -db {blastdb} \
    -query {workdir}/comb_{target}_hap.fasta -out {workdir}/comb_{target}_hap.tsv -outfmt 6 \
    -word_size 5 -max_target_seqs 1 -evalue 0.01"
    process = subprocess.run(cmd.split(), capture_output=True, text=True)

    # Handle errors
    if process.returncode != 0:
        logging.error(f"An error occurred while running the blastn command: {cmd}")
        logging.error(f"Command error: {process.stderr}")
        sys.exit(1)
    
    #Merge the blast results with the hap data and add additional columns
    blast_df = pd.read_csv(f'{workdir}/comb_{target}_hap.tsv', sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 
                                                                               'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])

    df = pd.merge(df.reset_index(), blast_df, how='right', left_on='blast_id', right_on='qseqid')
    df['genus'] = df.sseqid.str.split('_').str.get(0)
    df['specie'] = df.sseqid.str.split('_').str.get(1)
    df[f'ref_id_{target}'] = df['genus'] + '_' + df['specie']
    df['combUID'] = df.sseqid.str.split(':').str.get(1)

    #subset the dataframe to only the needed columns
    blast_df = df[[
        'sample_id','target', 'reads', 'total_reads', 'reads_fraction', 'consensus',
        f'ref_id_{target}', 'combUID', 'combUIDx', 'length','pident']].copy()
    blast_df['hap_id'] = df.apply(lambda x: x.combUID if x.pident == 100 else x.combUIDx, axis=1)

    return blast_df

def filter_blast(blast_df, target, filter_F1=10, filter_F2=10, filter_falciparum=False):

    # Filter out oversensitive haplotypes of Plasmodium falciparum for both P1 and P2
    logging.info('filtering blast output for falciparum')

    if filter_falciparum:
        if target == 'P1':
            df_f = blast_df[blast_df['combUID'].isin(['F1-0']) & (blast_df['reads']>= int(filter_F1))]
            df_x = blast_df[~blast_df['combUID'].isin(['F1-0'])]

        if target == 'P2':
            df_f = blast_df[blast_df['combUID'].isin(['F2-0']) & (blast_df['reads']>= int(filter_F2))]
            df_x = blast_df[~blast_df['combUID'].isin(['F2-0'])]

        blast_filt_df = pd.concat([df_f, df_x])
    else:
        blast_filt_df = blast_df.copy()

    return blast_filt_df

def haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_filt_df, workdir):

    """
    Calculate haplotype diversity for a given target and write the results to a file.

    Args:
        haplotype_df (pd.DataFrame): DataFrame containing haplotype data
        target (str): Target (P1 or P2)
        new_cols (list): List of new haplotype columns to add to the DataFrame
        hap_df (pd.DataFrame): DataFrame containing haplotype information
        blast_df (pd.DataFrame): Blast DataFrame
        outdir (str): Directory to write output file

    Returns:
        pd.DataFrame: DataFrame with haplotype diversity information

    """

    logging.info(f'determining the haplotype diversity for {target}')

    # #filter the input data to the current target
    hap_df_filt = hap_df[hap_df['target'] == target]

    #create the haplotype sequence dataframe
    haps_df = create_hap_data(hap_df_filt)
    hap_seq_df = pd.DataFrame({'haplotypes' :new_cols, 'sequences': haps_df.columns})

    #create a new haplotype dataframe
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

    #Add the combUIDs to the above dataframe
    merged_hap_df_blast = blast_filt_df[['consensus', 'hap_id', f'ref_id_{target}']].copy()
    merged_hap_df_blast.drop_duplicates(subset=['consensus', 'hap_id'], inplace=True)

    # Merge haplotype and combUID dataframes
    hap_div_df = pd.merge(
        left=merged_hap_df, left_on='sequences', right=merged_hap_df_blast, right_on='consensus', how='right')

    # Write output to file
    hap_div_df.to_csv(f'{workdir}/Plasmodium_haplotype_summary_for_{target}.tsv', sep='\t', index=False)

    return hap_div_df

def generate_haplotype_tree(target, hap_div_df, workdir, interactive_plotting=False):

    """
    Generates an alignment, tree files, and a bokeh alignment plot for the haplotypes of a given target.

    Args:
        target: The target to process.
        hap_div_df: The DataFrame containing the haplotype data.
        workdir: The path to the directory where the files will be saved.
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
    fig.savefig(f'{workdir}/haps_{target}_mafft.png')
    plt.close(fig)

    #View and save the Alignment using bokeh
    if interactive_plotting:

        from .iplot import view_alignment

        aln_fn = mafft_out_file
        aln = AlignIO.read(aln_fn, 'fasta')
        aln_view_fn = f'{workdir}/haps_{target}_mafft.html'
        
        view_alignment(aln, aln_view_fn, fontsize="9pt", plot_width=1200)      

def create_per_read_summary(blast_filt_df, target, outdir):

    """
    Generates a per-read summary for the given target.

    Parameters:
        blast_df (pd.DataFrame): The blast dataframe containing the reads.
        target (str): The target to generate the summary for.
        outdir (str): The directory to output the summary file to.

    Returns:
        pd.DataFrame: The summary dataframe.
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
        samples_df: DataFrame containing metadata for each sample
        merged_df: DataFrame containing summary outputs for each sample
        workdir: Directory to save the merged and exported dataframe

    Returns:
        A DataFrame containing the merged and exported results with additional columns for sample run and sample ID.
    """
    logging.info(f'merging the summary outputs with the sample metadata and exporting the merged dataframe to the work directory')

    # Merge the two dataframes and select only relevant columns
    df_merged = pd.merge(samples_df.set_index('sample_id'), merged_df, left_index=True, right_index=True, how='right')

    cols_to_keep = ['plate_id']

    if 'total_reads_P1' in df_merged.columns:
        cols_to_keep += ['total_reads_P1', 'ref_id_P1', 'haplotype_ID_P1', 'pident_P1',\
                    'reads_P1', 'hap_count_P1']

    if 'total_reads_P2' in df_merged.columns:
        cols_to_keep += ['total_reads_P2', 'ref_id_P2', 'haplotype_ID_P2', 'pident_P2',\
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

def process_results(haps_merged_df, filter_p1, filter_p2, workdir, outdir):

    """
    Read combined results summary TSV file and compute various metrics.

    Args:
        filter_p1 (str): Filter for Plasmodium P1 reads.
        filter_p2 (str): Filter for Plasmodium P2 reads.
        outdir (str): Directory containing the combined results summary TSV file.
        workdir (str): Working directory.

    Returns:
        pd.DataFrame: A pandas DataFrame containing various metrics for the samples.
    """

    logging.info(f'reading and processng the combined results summary TSV file and computing stats')

    def uniques(xs):
        return list(sorted(set(xi for x in xs for xi in x)))

    logging.info(f'reading results summary file and computing several metrics')

    df = pd.read_csv(f'{workdir}/combined_results_summary.tsv', sep='\t').set_index('sample_id')


    #create columns for fixing the read IDs
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

    #compute concordance and species
    # df["concordance"] = df[["reads_P1_name", "reads_P2_name"]].apply(uniques, axis=1).map(list)
    reads_cols = [col for col in ['reads_P1_name', 'reads_P2_name'] if col in df.columns]
    df['concordance'] = df[reads_cols].apply(uniques, axis=1).map(list)


    #spread out the plasmodium id
    df_all = pd.merge(df, pd.DataFrame(df['concordance'].values.tolist()).add_prefix('plasmodium_id_'), on=df.index)

    #set the index as sample_id
    df_all = df_all.rename(columns={'key_0': 'sample_id'}).set_index('sample_id')

    #create a final species column detailing what the species are and remove comments.
    df_all["plasmodium_species"] = df_all.filter(regex="^plasmodium_id_").apply(lambda x: ", ".join(sorted(filter(lambda y: pd.notnull(y) and y != "", x))), axis=1)
    df_all["plasmodium_species"] = df_all["plasmodium_species"].str.replace("'", "")

    #count the number of species per sample
    df_all['species_count'] = df['concordance'].apply(len)

    

    # Create Plasmodium status categories
    df_all['plasmodium_status'] = 'inconclusive'
    if 'P1_min' in df_all.columns and 'P2_min' in df_all.columns:
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'high_infection'
        df_all.loc[(df_all['P1_min'] == 0) & (df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'low_infection'
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] == 0), 'plasmodium_status'] = 'contradictory'

    elif 'P1_min' in df_all.columns:
        df_all.loc[(df_all['P1_min'] >= int(filter_p1)), 'plasmodium_status'] = 'contradictory'
     
    elif 'P2_min' in df_all.columns:
        df_all.loc[(df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'low_infection'

    

    # Create column for the presence of conflicts between P1 and P2
    if all(col in df_all.columns for col in ['reads_P1_name', 'reads_P2_name']):
        df_all['P1_P2_consistency'] = np.where(
            (df_all['reads_P1_name'].fillna('').apply(lambda x: len(set(x))) == 1) &
            (df_all['reads_P2_name'].fillna('').apply(lambda x: len(set(x))) == 1),
            'YES', 'NO')
    else:
        df_all['P1_P2_consistency'] = 'NO'
    
    # Create column for new haplotypes found
    # check if both P1_min_pident and P2_min_pident are present in the dataframe
    if all(col in df_all.columns for col in ['P1_min_pident', 'P2_min_pident']):
        df_all['new_haplotype'] = np.where(
            (df_all['P1_min_pident'].fillna(-1) < 100) | (df_all['P2_min_pident'].fillna(-1) < 100), 'YES', 'NO')
    # check if only P1_min_pident is present in the dataframe
    elif 'P1_min_pident' in df_all.columns:
        df_all['new_haplotype'] = np.where(
            df_all['P1_min_pident'].fillna(-1) < 100, 'YES', 'NO')
    # check if only P1_min_pident is present in the dataframe
    elif 'P2_min_pident' in df_all.columns:
        df_all['new_haplotype'] = np.where(
            df_all['P2_min_pident'].fillna(-1) < 100, 'YES', 'NO') 

    # if none of the two columns are present, set new_haplotype to NO for all rows
    else:
        df_all['new_haplotype'] = 'NO'
    

    # Remove unwanted characters from hap_ID_P1 and hap_ID_P2 columns
    for col in haps_merged_df['target'].unique():
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r'\[|\]|"', '', regex=True)
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r"'", "")
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r",", "\n")


    # Filter useful columns and save the results
    cols_to_keep = ['plate_id', 'plasmodium_species', 'plasmodium_status', 'species_count']
    if 'hap_ID_P1' in df_all.columns:
        cols_to_keep += ['hap_ID_P1', 'hap_count_P1', 'total_reads_P1']
    if 'hap_ID_P2' in df_all.columns:
        cols_to_keep += ['hap_ID_P2', 'hap_count_P2', 'total_reads_P2']                   
    if 'sample_supplier_name' in df_all.columns:
        cols_to_keep.insert(0, 'sample_supplier_name')
    cols_to_keep += ['new_haplotype', 'P1_P2_consistency']
    df_all_final = df_all[cols_to_keep]

    df_all_final.to_csv(f'{outdir}/plasmodium_predictions.tsv', sep='\t')

    

    return df_all_final

def generate_plots(meta_df_all, haps_merged_df, workdir, reference, interactive_plotting=False):
    """
    Generates plate and bar plots for the given metadata and haplotypes dataframes, and saves them in the specified
    directory.

    Parameters:
    meta_df_all (pandas.DataFrame): A dataframe containing the metadata results.
    haps_merged_df (pandas.DataFrame): A dataframe containing the haplotypes data.
    workdir (str): The path of the directory to save the plots in.
    reference (str): The reference of the run.
    interactive_plotting (bool): Whether to create interactive plots. Default is False.

    Returns:
    None
    """

    logging.info(f'generating plate and bar plots')

    # Create columns for sorting the dataframe
    meta_df_all['lims_row'] = meta_df_all.lims_well_id.str.slice(0,1)
    meta_df_all['lims_col'] = meta_df_all.lims_well_id.str.slice(1).astype(int)

    # Get the lims plate IDs
    limsplate = meta_df_all.lims_plate_id.unique()

    # Make categorical plots for each lims plate
    for lims_plate in limsplate:
        for target in haps_merged_df['target'].unique():
            if interactive_plotting:
                plot_plate_view(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(), 
                                f'{workdir}/plateview_for_{lims_plate}_{target}.html',
                                target, reference,
                                f'{lims_plate} Plasmodium positive samples for {target}')

        #Make numerical plots for each lims plate
        for target in haps_merged_df['target'].unique():
            plot_lims_plate(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(),
                            target, lims_plate,
                            f'{workdir}/plateview_heatmap_{lims_plate}_{target}.png', annot=True)

    # Make the bar plots
    plot_bar(meta_df_all, reference, f'{workdir}/bar_plots.png')


def plasm(args):

    """
    Run the plasm program to analyze ANOSPP QC data, including preparing input data and variables, running BLAST, 
    creating a dataframe, and generating plots.
    
    Parameters:
    args (argparse.Namespace): Namespace object containing command line arguments
    
    Returns:
    None
    """

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
    stats_df = prep_stats(args.stats)
    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)
    haps_merged_df = hard_filter_haplotypes(hap_df, args.filter_p1, args.filter_p2)

    # Check for presence of PLASM_TARGETS
    logging.info('checking for the presence of PLASM_TARGETS')
    if len(haps_merged_df['target'].unique()) < 1:
        logging.warning('Could not find both PLASM_TARGETS in hap_df')
        sys.exit(1)

    else:
        # Run BLAST and create haplotype tree for each target
        logging.info('running blast')
        df_list = []
        hap_output = []
        for target in haps_merged_df['target'].unique():          
            haplotype_df, new_cols = haplotype_summary(hap_df, target, args.workdir)
            blast_df = run_blast(haps_merged_df, target, args.workdir, args.blastdb)
            blast_filt_df = filter_blast(blast_df, target, args.filter_F1, args.filter_F2, args.filter_falciparum)
            hap_div_df = haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_filt_df, args.workdir)
            generate_haplotype_tree(target, hap_div_df, args.workdir, args.interactive_plotting)
            df_list.append(create_per_read_summary(blast_filt_df, target, args.workdir))
            hap_output.append(blast_filt_df)

        # Combine blast results and create merged dataframe for all targets
        logging.info('merging blast summary outputs')
        merged_df = pd.concat(df_list, axis=1)
        merge_and_export(samples_df, merged_df, args.workdir)
        df_all = process_results(haps_merged_df, args.filter_p1, args.filter_p2, args.workdir, args.outdir)
        merged_hap_df = pd.concat(hap_output, axis=0)[['sample_id', 'hap_id', 'target', 'consensus', 'pident']].copy().set_index('sample_id')
        merged_hap_df.to_csv(f'{args.outdir}/Plasmodium_haplotypes_for.tsv', sep='\t')

        # Merge sample and stats dataframes and generate plots
        logging.info('merging the samples(meta) dataframe with the stats dataframe and creating plots')
        meta_df_all = pd.merge(samples_df.set_index('sample_id'), df_all, left_index=True, right_index=True, how='left')
        generate_plots(meta_df_all, haps_merged_df, args.workdir, args.reference, args.interactive_plotting)

        logging.info('ANOSPP plasm complete')


def main():
    
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-b', '--blastdb', help='Blast db prefix. Default: ref_v1.0/plasmomito_P1P2_DB_v1.0', 
        default='ref_v1.0/plasmomito_P1P2_DB_v1.0', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='plasm')
    parser.add_argument('-w', '--workdir', help='Working directory. Default: work', default='work')
    parser.add_argument('-d', '--reference', help='Blast db prefix. Default: ref_v1.0', default='ref_v1.0')
    parser.add_argument('-c', '--readcutoff', help='Read cutoffs. Default: 10', default=10)
    parser.add_argument('-q', '--filter_p1', help='Plasmodium genus haplotype filter for P1. Default: 10', default=10)
    parser.add_argument('-r', '--filter_p2', help='Plasmodium genus haplotype filter for P2. Default: 10', default=10)
    parser.add_argument('-p', '--filter_F1', help='Plasmodium Falciparum main haplotype filter for P1. Default: 10', default=10)
    parser.add_argument('-l', '--filter_F2', help='Plasmodium Falciparum main haplotype filter for P2. Default: 10', default=10)
    parser.add_argument('-i', '--interactive_plotting', 
                            help='do interactive plotting', action='store_true', default=False)
    parser.add_argument('-f', '--filter_falciparum', 
                            help='Check for the highest occuring haplotypes of Plasmodium falciparum and filter', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')

    plasm(args)

if __name__ == '__main__':
    main()



#     # sort out dtypes
#     int_cols = [
#         'legacy_library_id', 
#         'manual_qc', 
#         'id_run', 
#         'position', 
#         'tag_index',
#         'id_sample_tmp',
#         'id_iseq_flowcell_tmp', 
#         'qc',
# #        'id_product',
#     ]

#     logging.info('refining dataframe')
#     # N.B., 'Int64' allows for missing values in pandas > 0.24
#     df = df.astype({k: 'Int64' for k in int_cols})
