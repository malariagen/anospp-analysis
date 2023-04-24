import glob
import argparse
import os
import subprocess
from subprocess import run
from collections import OrderedDict
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .util import *

###imports for bokeh
from Bio import AlignIO
from Bio import Phylo
from bokeh.plotting import output_file, save
# import panel as pn
import bokeh.plotting as bk
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Span, Range1d
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot, row, column
from bokeh.models.glyphs import Text, Rect
from bokeh.models.tools import HoverTool
from bokeh.transform import dodge

from bokeh.embed import components




def view_alignment(aln, fontsize="9pt", plot_width=800):
    """Bokeh sequence alignment view"""

    def get_colors(seqs):
        """make colors for bases in sequence"""
        text = [i for s in list(seqs) for i in s]
        clrs =  {'a':'red','t':'green','g':'orange','c':'blue','-':'white', 'n':'black'}
        colors = [clrs[i] for i in text]
        return colors

    #make sequence and id lists from the aln object
    seqs = [rec.seq for rec in (aln)]
    ids = [rec.id for rec in aln]    
    text = [i for s in list(seqs) for i in s]
    colors = get_colors(seqs)    
    N = len(seqs[0])
    S = len(seqs)    
    width = .4

    x = np.arange(1,N+1)
    y = np.arange(0,S,1)
    #creates a 2D grid of coords from the 1D arrays
    xx, yy = np.meshgrid(x, y)
    #flattens the arrays
    gx = xx.ravel()
    gy = yy.flatten()
    #use recty for rect coords with an offset
    recty = gy+.5
    h= 1/S
    #now we can create the ColumnDataSource with all the arrays
    source = ColumnDataSource(dict(x=gx, y=gy, recty=recty, text=text, colors=colors))
    plot_height = len(seqs)*15+50
    x_range = Range1d(0,N+1, bounds='auto')
    if N>100:
        viewlen=100
    else:
        viewlen=N
    #view_range is for the close up view
    view_range = (0,viewlen)
    tools="xpan, xwheel_zoom, reset, save"

    #entire sequence view (no text, with zoom)
    p = figure(title=None, plot_width= plot_width, plot_height=50,
               x_range=x_range, y_range=(0,S), tools=tools,
               min_border=0, toolbar_location='below')
    rects = Rect(x="x", y="recty",  width=1, height=1, fill_color="colors",
                 line_color=None, fill_alpha=0.6)
    p.add_glyph(source, rects)
    p.yaxis.visible = False
    p.grid.visible = False  

    #sequence text view with ability to scroll along x axis
    p1 = figure(title=None, plot_width=plot_width, plot_height=plot_height,
                x_range=view_range, y_range=ids, tools="xpan,reset",
                min_border=0, toolbar_location='below')#, lod_factor=1)          
    glyph = Text(x="x", y="y", text="text", text_align='center',text_color="black",
                text_font="monospace",text_font_size=fontsize)
    rects = Rect(x="x", y="recty",  width=1, height=1, fill_color="colors",
                line_color=None, fill_alpha=0.4)
    p1.add_glyph(source, glyph)
    p1.add_glyph(source, rects)

    p1.grid.visible = False
    p1.xaxis.major_label_text_font_style = "bold"
    p1.yaxis.minor_tick_line_width = 0
    p1.yaxis.major_tick_line_width = 0

    p = gridplot([[p],[p1]], toolbar_location='below')
    return p

def plot_lims_plate(df, target, plate, fname, reference, title=None, annot=True, cmap='coolwarm', center=None):
    """
    Plot a heatmap of the total read count for a given target on a plate.

    Args:
        df (pandas.DataFrame): A dataframe containing the read counts and positions on the plate.
        target (str): The name of the target (e.g. 'P1', 'P2').
        title (str): The title of the plot. Default is None.
        plate (str): The name of the plate. Default is None.
        annot (bool): Whether to annotate the heatmap cells with their values. Default is True.
        cmap (str or colormap): The colormap to use. Default is 'coolwarm'.
        center (float): The value at which to center the colormap. Default is None.
        filename (str): The name of the file to save the plot to. Default is None.

    Returns:
        None
    """
    #load colors
    if not os.path.isfile(f'{reference}/species_colours.csv'):
        logging.warning('No colors defined for plotting.')
    else:
        colors = pd.read_csv(f'{reference}/species_colours.csv')
        colors_dict = dict(zip(colors['species'], colors['color']))


    

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    col = f'total_reads_{target}'
    sns.heatmap(df.pivot(index = 'lims_row', columns='lims_col', values= col), annot=annot, cmap=colors_dict, ax=ax, center=center)
    ax.set_title(f"Total {target} reads for {plate}")
    ax.hlines([i * 2 for i in range(9)], 0, 24, colors='k')
    ax.vlines([j * 2 for j in range(13)], 0, 16, colors='k')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_plate_view(df, fname, target, reference, title=None):

    """
    Plots a plate map for a given plate and Plasmodium type.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    P (str): The Plasmodium type to plot the total reads for.
    title (str): The title for the plot. Default is None.
    plate (str): The name of the plate. Default is None.
    annot (bool): Whether to annotate the heatmap with the values. Default is True.
    cmap (str): The color map to use for the heatmap. Default is 'coolwarm'.
    center (float): The center value for the color map. Default is None.

    Returns:
    None.
    """

 
    # set the output filename
    output_file(fname)

    #extract the column and generate the row values
    cols = list(map(str, sorted(df.lims_row.unique().tolist())))
    rows = [str(x) for x in range(1, 25)]
    df["species_count"] = df["species_count"].astype(str)
    df["row"] = df["lims_col"].astype(str)
    df["col"] = df["lims_row"].astype(str)

    #remove all NaNs
    df = df[df.species_count != "nan"]

    #load the datframe into the source
    source = ColumnDataSource(df)

    #set up the figure
    p = figure(width=1300, height=600, title=title,
               x_range=rows, y_range=list(reversed(cols)), toolbar_location=None, tools=[HoverTool(), 'pan', 'wheel_zoom', 'reset'])

    # add grid lines
    for v in range(len(rows)):
        vline = Span(location=v, dimension='height', line_color='black')
        p.renderers.extend([vline])

    for h in range(len(cols)):
        hline = Span(location=h, dimension='width', line_color='black')
        p.renderers.extend([hline])

    #add the rectangles
    # cmap = OrderedDict([("Plasmodium_falciparum", "red"), ("Plasmodium_vivax", "blue"), ("Plasmodium_malariae", "green")])
    #load colors
    if not os.path.isfile(f'{reference}/species_colours.csv'):
        logging.warning('No colors defined for plotting.')
    else:
        colors = pd.read_csv(f'{reference}/species_colours.csv')
        cmap = dict(zip(colors['species'], colors['color']))
        # print(cmap)
    p.rect("row", "col", 0.95, 0.95, source=source, fill_alpha=.9, legend_field="plasmodium_species",
           color=factor_cmap('plasmodium_species', palette=list(cmap.values()), factors=list(cmap.keys())))

    #add the species count text for each field
    text_props = {"source": source, "text_align": "left", "text_baseline": "middle"}
    x = dodge("row", -0.4, range=p.x_range)
    if target == 'P1':
        r = p.text(x=x, y="col", text="hap_ID_P1", **text_props, )

    else:
        r = p.text(x=x, y="col", text="hap_ID_P2", **text_props, )
    r.glyph.text_font_size="10px"
    r.glyph.text_font_style="bold"

    #set up the hover value
    p.add_tools(HoverTool(tooltips=[
        ("sample id", "@{Source_sample}"),
        ("Parasite species", "@plasmodium_species"),
        ("Detection confidence", "@plasmodium_status"),
        ("P1 & P2 consistency", "@P1_P2_consistency"),
        ("P1 haplotype ID", "@hap_ID_P1"),
        ("Total P1 read count", "@total_reads_P1"),
        ("P2 haplotype ID", "@hap_ID_P2"),
        ("Total P2 read count", "@total_reads_P2"),
    ]))

    #set up the rest of the figure and save the plot
    p.outline_line_color = 'black'
    p.grid.grid_line_color = None
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = "vertical"
    p.legend.click_policy="hide"
    p.add_layout(p.legend[0], 'right') 
    save(p)



    



def plot_bar(df, fname):
    """
    Plots bar charts of plasmodium species counts, grouped by plate ID and plasmodium status.

    Args:
    - df: pandas DataFrame containing the plasmodium species and status counts for each plate
    - fname: str, file name to save the plot

    Returns:
    - None
    """
    # Drop rows with missing values in 'plasmodium_species' or 'plasmodium_status'
    data = df.dropna(subset=['plasmodium_species', 'plasmodium_status'])

    # Group data by plasmodium species, status, and plate ID and count the occurrences
    plasmodium_count = pd.DataFrame({'count': data.groupby(["plasmodium_species", "plasmodium_status", "lims_plate_id"]).size()}).reset_index()

    # Set up the plots
    plt.figure(figsize=(8, 8))
    sns.set_context(rc={'patch.linewidth': 0.5})
    ax = sns.catplot(x="plasmodium_species", y="count", row="lims_plate_id",
                     col="plasmodium_status", data=plasmodium_count, kind="bar",
                     estimator=sum, facet_kws={'legend_out': True})

    # Customize plot labels
    ax.set_xticklabels(rotation=40, ha="right", fontsize=9)
    ax.set_xlabels('Plasmodium species predictions', fontsize=12)
    ax.set_ylabels('Species counts', fontsize=12)
    plt.tight_layout()

    #save the plot to file
    ax.savefig(fname, dpi=300, bbox_inches='tight')



def process_haplotypes(hap_df, comb_stats_df, filter_p1, filter_p2):

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

    logging.info('process the haplotype values')
    
    # pull out the haplotypes that meet the filter_value
    p1_filter = (hap_df["target"] == "P1") & (hap_df["reads"] >= int(filter_p1))
    p2_filter = (hap_df["target"] == "P2") & (hap_df["reads"] >= int(filter_p2))
    filtered_hap_df =  hap_df[p1_filter | p2_filter]

    #Filter out columns that have no recorded samples
    haps_merged_df = filtered_hap_df.loc[:, (filtered_hap_df != 0).any(axis=0)]

    return(haps_merged_df)

def create_hap_data(hap_df):
    """
    Create a dataframe with haplotype and reads/sample stats from a haplotype dataframe.

    Args:
        hap_df (pandas.DataFrame): The haplotype dataframe.

    Returns:
        pandas.DataFrame: A dataframe with haplotype and reads/sample stats.
    """

    # Create pivot table data for dada2 haplotypes using the hap_df data
    hap_data_filt = hap_df[['sample_id', 'consensus', 'reads']].copy()
    haps_df = hap_data_filt.pivot_table(values='reads', index='sample_id', columns='consensus')
    haps_df.fillna(0, inplace=True)

    # Remove 'consensus' header
    haps_df.columns.name = None

    # Move 'sample_id' index to column
    haps_df = haps_df.reset_index()

    # Filter out columns that have no recorded samples
    haps_df = haps_df.loc[:, (haps_df != 0).any(axis=0)]

    # Set 'sample_id' as index and convert the values to integer
    haps_df = haps_df.set_index('sample_id').astype(int)

    return haps_df

def haplotype_summary(hap_df: pd.DataFrame, target: str, workdir: str) -> pd.DataFrame:
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

    logging.info(f"Processing the haplotype data for {target} and generating a table summary.")

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

def run_blast(hap_data, target, workdir, blastdb, filter_F1=10, filter_F2=10):

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
    df = df[[
        'sample_id','target', 'reads', 'total_reads', 'reads_fraction', 'consensus',
        f'ref_id_{target}', 'combUID', 'combUIDx', 'length','pident']].copy()
    df['hap_id'] = df.apply(lambda x: x.combUID if x.pident == 100 else x.combUIDx, axis=1)

    #Filter out oversensitve haplotypes of Plasmodium falciparum for both P1 and P2
    if target == 'P1':
        df_f = df[df['combUID'].isin(['F1-0']) & (df['reads']>= int(filter_F1))]
        df_x = df[~df['combUID'].isin(['F1-0'])]

    if target == 'P2':
        df_f = df[df['combUID'].isin(['F2-0']) & (df['reads']>= int(filter_F2))]
        df_x = df[~df['combUID'].isin(['F2-0'])]

    blast_df = pd.concat([df_f, df_x])

    return blast_df

def haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_df, workdir):

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
    merged_hap_df_blast = blast_df[['consensus', 'hap_id', f'ref_id_{target}']].copy()
    merged_hap_df_blast.drop_duplicates(subset=['consensus', 'hap_id'], inplace=True)

    # Merge haplotype and combUID dataframes
    hap_div_df = pd.merge(
        left=merged_hap_df, left_on='sequences', right=merged_hap_df_blast, right_on='consensus', how='right')

    # Write output to file
    hap_div_df.to_csv(f'{workdir}/Plasmodium_haplotype_summary_for_{target}.tsv', sep='\t', index=False)

    return hap_div_df

def generate_haplotype_tree(target: str, hap_div_df: pd.DataFrame, workdir: str):

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
    aln_fn = mafft_out_file
    aln = AlignIO.read(aln_fn, 'fasta')
    output_file(filename=f'{workdir}/haps_{target}_mafft.html', title="Static HTML file")
    p = view_alignment(aln, plot_width=1200)

    save(p)

def create_per_read_summary(blast_df: pd.DataFrame, target: str, outdir: str) -> pd.DataFrame:

    """
    Generates a per-read summary for the given target.

    Parameters:
        blast_df (pd.DataFrame): The blast dataframe containing the reads.
        target (str): The target to generate the summary for.
        outdir (str): The directory to output the summary file to.

    Returns:
        pd.DataFrame: The summary dataframe.
    """

    # Create a dataframe with the relevant stats
    df_sum = blast_df.groupby(['sample_id', 'total_reads']).agg(
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

def merge_and_export(samples_df: pd.DataFrame, merged_df: pd.DataFrame, workdir: str) -> pd.DataFrame:

    """
    Merge summary outputs with metadata and export the resulting dataframe to a specified directory.

    Args:
        samples_df: DataFrame containing metadata for each sample
        merged_df: DataFrame containing summary outputs for each sample
        workdir: Directory to save the merged and exported dataframe

    Returns:
        A DataFrame containing the merged and exported results with additional columns for sample run and sample ID.
    """
    # Merge the two dataframes and select only relevant columns
    df_merged = pd.merge(samples_df.set_index('sample_id'), merged_df, left_index=True, right_index=True, how='right')
    df_final = df_merged[['sample_supplier_name', 'plate_id', 'total_reads_P1',
                          'ref_id_P1', 'haplotype_ID_P1', 'pident_P1', 'reads_P1',
                          'hap_count_P1', 'total_reads_P2', 'ref_id_P2', 'haplotype_ID_P2',
                          'pident_P2', 'reads_P2', 'hap_count_P2']].copy()
    
    # Add columns for sample ID
    df_final.index.name = 'sample_id'
    
    # Export the merged dataframe to a TSV file
    file_name = f'{workdir}/combined_results_summary.tsv'
    df_final.to_csv(file_name, sep='\t')
    
    return df_final

def process_results(filter_p1, filter_p2, workdir, outdir):

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

    def uniques(xs):
        return list(sorted(set(xi for x in xs for xi in x)))

    logging.info(f'reading results summary file and computing several metrics')

    df = pd.read_csv(f'{workdir}/combined_results_summary.tsv', sep='\t').set_index('sample_id')

    #create columns for fixing the read IDs
    for col in PLASM_TARGETS:
        df[f'reads_{col}_name'] = df[f'ref_id_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else '')
        df[f'reads_{col}_fixed'] = df[f'reads_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else [0])
        df[f'pident_{col}_fixed'] = df[f'pident_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else [0])

    for col in PLASM_TARGETS:
        df[f"{col}_min"] = df[f"reads_{col}_fixed"].apply(lambda x: min(int(y) for y in x) if x != ["0"] else 0)
        df[f"{col}_max"] = df[f"reads_{col}_fixed"].apply(lambda x: max(int(y) for y in x) if x != ["0"] else 0)
        df[f"{col}_avg"] = df[f"reads_{col}_fixed"].apply(lambda x: sum(int(y) for y in x) / len(x) if x != ["0"] else 0)
        df[f"{col}_min_pident"] = df[f"pident_{col}_fixed"].apply(lambda x: min(float(y) for y in x) if x != ["0"] else 0)
        df[f'hap_ID_{col}'] = df[f'haplotype_ID_{col}'].apply(lambda d: d.strip('][').split(', ') if isinstance(d, str) else '')

    #compute concordance and species
    df["concordance"] = df[["reads_P1_name", "reads_P2_name"]].apply(uniques, axis=1).map(list)

    #spread out the plasmodium id
    df_all = pd.merge(df, pd.DataFrame(df['concordance'].values.tolist()).add_prefix('plasmodium_id_'), on=df.index)

    #set the index as sample_id
    df_all = df_all.rename(columns={'key_0': 'sample_id'}).set_index('sample_id')

    #create a final species column detailing what the species are and remove comments.
    df_all["plasmodium_species"] = df_all.filter(regex="^plasmodium_id_").astype(str).apply(lambda x: ", ".join(sorted(filter(None, x))), axis=1)
    df_all["plasmodium_species"] = df_all["plasmodium_species"].str.replace("'", "")

    #count the number of species per sample
    df_all['species_count'] = df['concordance'].apply(len)

    # Create Plasmodium status categories
    df_all['plasmodium_status'] = ''
    df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'high_infection'
    df_all.loc[(df_all['P1_min'] == 0) & (df_all['P2_min'] >= int(filter_p2)), 'plasmodium_status'] = 'low_infection'
    df_all.loc[(df_all['P1_min'] >= int(filter_p1)) & (df_all['P2_min'] == 0), 'plasmodium_status'] = 'contradictory'

    # Create column for the presence of conflicts between P1 and P2
    df_all['P1_P2_consistency'] = np.where((df_all['reads_P1_name'].apply(lambda x: len(set(x))) == 1) &
                                            (df_all['reads_P2_name'].apply(lambda x: len(set(x))) == 1), 'YES', 'NO')
    
    # Create column for new haplotypes found
    df_all['new_haplotype'] = np.where(
        (df_all['P1_min_pident'] < 100) | (df_all['P2_min_pident'] < 100), 'YES', 'NO')

    # Remove unwanted characters from hap_ID_P1 and hap_ID_P2 columns
    for col in PLASM_TARGETS:
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r'\[|\]|"', '', regex=True)
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r"'", "")
        df_all[f'hap_ID_{col}'] = df_all[f'hap_ID_{col}'].astype(str).str.replace(r",", "\n")

    # Filter useful columns and save the results
    cols_to_keep = ['sample_supplier_name', 'plate_id', 'plasmodium_species', 'species_count', 'plasmodium_status',
                    'hap_ID_P1', 'hap_count_P1', 'total_reads_P1', 'hap_ID_P2', 'hap_count_P2', 'total_reads_P2',
                    'new_haplotype', 'P1_P2_consistency']

    df_all[cols_to_keep].to_csv(f'{outdir}/plasmodium_predictions.tsv', sep='\t')

    return df_all


def generate_plots(meta_df_all, haps_merged_df, workdir, reference):
    """
    Generate plate and bar plots for the given meta_results dataframe and run name.

    Parameters:
    meta_results (pandas.DataFrame): dataframe containing the metadata results
    run (str): name of the current run

    Returns:
    None
    """
    # Create columns for sorting the dataframe
    meta_df_all['lims_row'] = meta_df_all.lims_well_id.str.slice(0,1)
    meta_df_all['lims_col'] = meta_df_all.lims_well_id.str.slice(1).astype(int)

    # Get the lims plate IDs
    limsplate = meta_df_all.lims_plate_id.unique()

    # # Set up the color map
    # cmap = {"": "#FFFFFF"}
    # color_l = list(mcolors.CSS4_COLORS)
    # for i, species in enumerate(sorted(meta_df_all['plasmodium_species'].astype(str).unique())):
    #     cmap[species] = mcolors.CSS4_COLORS[color_l[-i]]

    # print(cmap)

    # Make categorical plots for each lims plate
    for lims_plate in limsplate:
        for i in haps_merged_df['target'].unique():
            plot_plate_view(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(), 
                            f'{workdir}/plateview_for_{lims_plate}_{i}.html',
                            i, reference,
                            f'{lims_plate} Plasmodium positive samples')

        # #Make numerical plots for each lims plate
        # for i in haps_merged_df['target'].unique():
        #     plot_lims_plate(meta_df_all[meta_df_all.lims_plate_id == lims_plate].copy(),
        #                     i, lims_plate,
        #                     f'{workdir}/plateview_heatmap_{lims_plate}_{i}.png',
        #                     f'{reference}/species_colours.csv', annot=False)

    # # Make the bar plots
    # plot_bar(meta_df_all, f'{workdir}/bar_plots.png')



def plasm(args):

    setup_logging(verbose=args.verbose)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    logging.info('ANOSPP QC data import started')
    hap_df = prep_hap(args.haplotypes)
    # plasm_hap_df = hap_df.loc[hap_df.target.isin(PLASM_TARGETS)]  
    logging.info('## prepare input data and variables')
    samples_df = prep_samples(args.manifest)
    stats_df = prep_stats(args.stats)
    comb_stats_df = combine_stats(stats_df, hap_df, samples_df)
    # run_id = '_'.join(str(value) for value in comb_stats_df.run_id.unique())
    haps_merged_df = process_haplotypes(hap_df, comb_stats_df, args.filter_p1, args.filter_p2)

    logging.info('Checking for the presence of PLASM_TARGETS')
    if len(haps_merged_df['target'].unique()) < 1:
        logging.warning('Could not find both PLASM_TARGETS in hap_df')
        sys.exit(1)

    else:
        logging.info('## run blast')
        df_list = []
        hap_output = []
        for target in haps_merged_df['target'].unique():          
            haplotype_df, new_cols = haplotype_summary(hap_df, target, args.workdir)
            blast_df = run_blast(haps_merged_df, target, args.workdir, args.blastdb, args.filter_F1, args.filter_F2)
            hap_div_df = haplotype_diversity(haplotype_df, target, new_cols, hap_df, blast_df, args.workdir)
            generate_haplotype_tree(target, hap_div_df, args.workdir)
            df_list.append(create_per_read_summary(blast_df, target, args.workdir))
            hap_output.append(blast_df)

        logging.info('## create a dataframe used the merged summary outputs')
        merged_df = pd.concat(df_list, axis=1)
        merge_and_export(samples_df, merged_df, args.workdir)
        df_all = process_results(args.filter_p1, args.filter_p2, args.workdir, args.outdir)
        merged_hap_df = pd.concat(hap_output, axis=0)[['sample_id', 'hap_id', 'target', 'consensus', 'pident']].copy().set_index('sample_id')
        merged_hap_df.to_csv(f'{args.outdir}/Plasmodium_haplotypes_for.tsv', sep='\t')

        logging.info('merging the samples(meta) dataframe with the stats dataframe and creating plots')
        meta_df_all = pd.merge(samples_df.set_index('sample_id'), df_all, left_index =True, right_index=True, how='left')
        generate_plots(meta_df_all, haps_merged_df, args.workdir, args.reference)

        logging.info('## completed the plasm program!!!')


def main():
    
    parser = argparse.ArgumentParser("QC for ANOSPP sequencing data")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file', required=True)
    parser.add_argument('-m', '--manifest', help='Samples manifest tsv file', required=True)
    parser.add_argument('-s', '--stats', help='DADA2 stats tsv file', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory. Default: qc', default='plasm')
    parser.add_argument('-w', '--workdir', help='Working directory. Default: work', default='work')
    parser.add_argument('-b', '--blastdb', help='Blast db prefix. Default: ref_v1.0/plasmomito_P1P2_DB_v1.0', default='ref_v1.0/plasmomito_P1P2_DB_v1.0')
    parser.add_argument('-d', '--reference', help='Blast db prefix. Default: ref_v1.0', default='ref_v1.0')
    parser.add_argument('-c', '--readcutoff', help='Read cutoffs. Default: 10', default=10)
    parser.add_argument('-q', '--filter_p1', help='Plasmodium genus haplotype filter for P1. Default: 10', default=10)
    parser.add_argument('-r', '--filter_p2', help='Plasmodium genus haplotype filter for P2. Default: 10', default=10)
    parser.add_argument('-p', '--filter_F1', help='Plasmodium Falciparum main haplotype filter for P1. Default: 10', default=10)
    parser.add_argument('-f', '--filter_F2', help='Plasmodium Falciparum main haplotype filter for P2. Default: 10', default=10)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    plasm(args)

if __name__ == '__main__':
    main()



