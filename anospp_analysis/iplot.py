###imports for bokeh
from bokeh.plotting import output_file, save
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, Span, Range1d
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot
from bokeh.models.glyphs import Text, Rect
from bokeh.models.tools import HoverTool
from bokeh.transform import dodge

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anospp_analysis.util import *


def view_alignment(aln, aln_view_fn, fontsize='9pt', plot_width=1200):
    '''Bokeh sequence alignment view'''

    def get_colors(seqs):
        '''make colors for bases in sequence'''
        text = [i for s in list(seqs) for i in s]
        clrs =  {'a':'red','t':'green','g':'orange','c':'blue','-':'white', 'n':'black'}
        colors = [clrs[i] for i in text]
        return colors

    output_file(filename=aln_view_fn, title='Static HTML file')

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
    tools='xpan, xwheel_zoom, reset, save'

    #entire sequence view (no text, with zoom)
    p = figure(title=None, width = plot_width, height=50,
               x_range=x_range, y_range=(0,S), tools=tools,
               min_border=0, toolbar_location='below')
    rects = Rect(x='x', y='recty',  width=1, height=1, fill_color='colors',
                 line_color=None, fill_alpha=0.6)
    p.add_glyph(source, rects)
    p.yaxis.visible = False
    p.grid.visible = False  

    #sequence text view with ability to scroll along x axis
    p1 = figure(title=None, width=plot_width, height=plot_height,
                x_range=view_range, y_range=ids, tools='xpan,reset',
                min_border=0, toolbar_location='below')#, lod_factor=1)          
    glyph = Text(x='x', y='y', text='text', text_align='center',text_color='black',
                text_font='monospace',text_font_size=fontsize)
    rects = Rect(x='x', y='recty',  width=1, height=1, fill_color='colors',
                line_color=None, fill_alpha=0.4)
    p1.add_glyph(source, glyph)
    p1.add_glyph(source, rects)

    p1.grid.visible = False
    p1.xaxis.major_label_text_font_style = 'bold'
    p1.yaxis.minor_tick_line_width = 0
    p1.yaxis.major_tick_line_width = 0

    p = gridplot([[p],[p1]], toolbar_location='below')
    save(p)

def plot_plate_view(df, out_fn, reference_path, title=None):

    '''
    Plots a plate map for a given plate and Plasmodium type.

    Args:
    - df (pandas.DataFrame): DataFrame to plot.
    - out_fn (str): name of the file to save the plot
    - reference_path (str): path to the reference directory
    - title (str): title for the plot. Default is None.

    Returns:
    None.
    '''

    assert os.path.isdir(reference_path), f'{reference_path} reference directory does not exist'

    # set the output filename
    output_file(out_fn)

    #extract the column and generate the row values
    cols = list('ABCDEFGHIJKLMNOP')
    rows = [str(x) for x in range(1, 25)]
    df = df.reset_index(drop=False)
    df['col'] = df['lims_well_id'].str[0]
    df['row'] = df['lims_well_id'].str[1:]
    
    # display values
    df['P1_hapids_disp'] = df['P1_hapids_pass'].str.replace(',.*', '...', regex=True)
    df['P2_hapids_disp'] = df['P2_hapids_pass'].str.replace(',.*', '...', regex=True)
    df['comb_hapids_disp'] = df['P1_hapids_disp'] + '\n' + df['P2_hapids_disp']

    #load the dataframe into the source
    source = ColumnDataSource(df)

    #set up the figure
    p = figure(
        width=1300,
        height=600,
        title=title,
        x_range=rows,
        y_range=list(reversed(cols)),
        toolbar_location=None,
        tools=[HoverTool(), 'pan', 'wheel_zoom', 'reset']
        )

    # add grid lines
    for v in range(len(rows)):
        line_width = 2 if (v % 2 == 0) else 1
        vline = Span(location=v, dimension='height', line_color='black', line_width=line_width)
        p.renderers.extend([vline])

    for h in range(len(cols)):
        line_width = 2 if (h % 2 == 0) else 1
        hline = Span(location=h, dimension='width', line_color='black', line_width=line_width)
        p.renderers.extend([hline])

    #load colors
    if not os.path.isfile(f'{reference_path}/species_colours.csv'):
        logging.warning('no colors defined for plotting')
        cmap = {}
    else:
        colors = pd.read_csv(f'{reference_path}/species_colours.csv')
        cmap = dict(zip(colors['species'], colors['color']))

    
    for index, row in df.iterrows():
        # Assign grey color to data with more than one species
        if len(row['plasmodium_species'].split(',')) > 1:
            cmap[row['plasmodium_species']] = '#cfcfcf'
        # Assign white color to data with no species
        elif len(row['plasmodium_species']) == 0:
            cmap[row['plasmodium_species']] = '#ffffff'

    #add the rectangles
    p.rect(
        'row',
        'col',
        0.95,
        0.95,
        source=source,
        fill_alpha=.9,
        legend_field='plasmodium_species',
        color=factor_cmap('plasmodium_species', palette=list(cmap.values()), factors=list(cmap.keys()))
        )

    #add the species count text for each field
    text_props = {'source': source, 'text_align': 'left', 'text_baseline': 'middle'}
    x = dodge('row', -0.4, range=p.x_range)
    r = p.text(x=x, y='col', text='comb_hapids_disp', **text_props)
    r.glyph.text_font_size = '10px'
    r.glyph.text_font_style = 'bold'

    #set up the hover value
    p.add_tools(HoverTool(tooltips=[
        ('sample id', '@{sample_id}'),
        ('Parasite species', '@plasmodium_species'),
        ('Detection status', '@plasmodium_detection_status'),
        ('P1 total reads', '@P1_reads_total'),
        ('P1 QC pass reads', '@P1_reads_pass'),
        ('P1 QC pass haplotype IDs', '@P1_hapids_pass'),
        ('P1 reads per QC pass haplotype', '@P1_hapids_pass_reads'),
        ('P1 species assignments for pass haplotypes', '@P1_species_assignments_pass'),
        ('P1 contamination haplotype IDs', '@P1_hapids_contam'),
        ('P1 reads per contamination haplotype', '@P1_hapids_contam_reads'),
        ('P1 low coverage haplotype IDs', '@P1_hapids_locov'),
        ('P1 reads per low coverage haplotype', '@P1_hapids_locov_reads'),
        ('P2 total reads', '@P2_reads_total'),
        ('P1 QC pass reads', '@P2_reads_pass'),
        ('P2 QC pass haplotype IDs', '@P2_hapids_pass'),
        ('P2 reads per QC pass haplotype', '@P2_hapids_pass_reads'),
        ('P2 species assignments for pass haplotypes', '@P2_species_assignments_pass'),
        ('P2 contamination haplotype IDs', '@P2_hapids_contam'),
        ('P2 reads per contamination haplotype', '@P2_hapids_contam_reads'),
        ('P2 low coverage haplotype IDs', '@P2_hapids_locov'),
        ('P2 reads per low coverage haplotype', '@P2_hapids_locov_reads'),
    ]))

    #set up the rest of the figure and save the plot
    p.outline_line_color = 'black'
    p.grid.grid_line_color = None
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = 'vertical'
    p.legend.click_policy='hide'
    p.add_layout(p.legend[0], 'right') 
    save(p)