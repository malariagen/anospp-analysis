# anospp-analysis

Python package for ANOSPP data analysis

ANOSPP is the multiplexed amplicon sequencing assay for Anopheles mosquito species identification and Plasmodium detection. This repository contains the code for analysis of the sequencing results pre-processed with [nf-core ampliseq](https://nf-co.re/ampliseq) pipeline. 

## Usage

Key analysis steps are implemented as standalone scripts:
- `anospp-prep` takes DADA2 output files and targets primer sequences, demultiplexes the amplicons and yields haplotypes table
- `anospp-qc` takes haplotypes table, DADA2 stats table and samples manifest and produces QC plots 

## Development

### Setup

Installation is hybrid with conda + poetry:
```
git clone git@github.com:malariagen/anospp-analysis.git
cd anospp-analysis
mamba env create -f environment.yml
conda activate anospp_analysis
poetry install
```

Activate development environment:
```
poetry shell
```

### Usage & testing

The code in this repository can be accessed via wrapper scripts:
```
anospp-qc \
    --haplotypes test_data/haplotypes.tsv \
    --samples test_data/samples.csv \
    --stats test_data/stats.tsv \
    --outdir test_data/qc
```

Besides, individual components are available as a python API:
```
$ python
>>> from anospp_analysis.util import *
>>> PLASM_TARGETS
['P1', 'P2']
```

Automated testing & CI

### Adding Python deps

Introducing python dependencies should be done via poetry:
```
poetry add package_name
``` 
This should update both `pyproject.toml` and `poetry.lock` files

If the package should be used in development environment only, use
```
poetry add package_name --dev
```

### Adding non-Python deps

Introducing non-python dependencies should be done via conda: edit `environment.yml`, 
then re-create the conda environment and poetry deps:
```
mamba env create -f environment.yml
conda activate anospp_analysis
poetry install
```

Changes in conda environment might also introduce changes to the python installation, 
in which case one should update poetry lock file
```
poetry lock
```

## Release checklist

While in dev branch
- test functionality (TODO CI)
- bump version in `pyproject.toml`

Then
- merge into master
- github release
- pypi release

Conda recipe update `conda.recipe/meta.yaml`:
- check deps vs `environment.yaml` and `pyproject.toml`
- bump version  
- update sha256 
- test conda recipe with `bioconda-utils build --git-range master` vs bioconda-recipes
