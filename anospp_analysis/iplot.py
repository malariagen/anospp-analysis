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

from .util import *


def view_alignment(aln, aln_view_fn, fontsize="9pt", plot_width=1200):
    """Bokeh sequence alignment view"""

    def get_colors(seqs):
        """make colors for bases in sequence"""
        text = [i for s in list(seqs) for i in s]
        clrs =  {'a':'red','t':'green','g':'orange','c':'blue','-':'white', 'n':'black'}
        colors = [clrs[i] for i in text]
        return colors

    output_file(filename=aln_view_fn, title="Static HTML file")

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
    p = figure(title=None, width = plot_width, height=50,
               x_range=x_range, y_range=(0,S), tools=tools,
               min_border=0, toolbar_location='below')
    rects = Rect(x="x", y="recty",  width=1, height=1, fill_color="colors",
                 line_color=None, fill_alpha=0.6)
    p.add_glyph(source, rects)
    p.yaxis.visible = False
    p.grid.visible = False  

    #sequence text view with ability to scroll along x axis
    p1 = figure(title=None, width=plot_width, height=plot_height,
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
    save(p)

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

    #load colors
    if not os.path.isfile(f'{reference}/species_colours.csv'):
        logging.warning('No colors defined for plotting.')
        cmap = {}
    else:
        colors = pd.read_csv(f'{reference}/species_colours.csv')
        cmap = dict(zip(colors['species'], colors['color']))

    # Assign grey color to data with more than one species
    for index, row in df.iterrows():
        if len(row['plasmodium_species'].split(',')) > 1:
            cmap[row['plasmodium_species']] = '#cfcfcf'

    #add the rectangles
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