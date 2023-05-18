# anospp-analysis

Python package for ANOSPP data analysis

ANOSPP is the multiplexed amplicon sequencing assay for Anopheles mosquito species identification and Plasmodium detection. This repository contains the code for analysis of the sequencing results pre-processed with [nf-core ampliseq](https://nf-co.re/ampliseq) pipeline.

## Installation

For released version

```bash
conda install -c bioconda anospp-analysis
```

For development setup, see instructions below

## Usage

Key analysis steps are implemented as standalone scripts:

- `anospp-prep` takes DADA2 output files and targets primer sequences, demultiplexes the amplicons and yields haplotypes table
- `anospp-qc` takes haplotypes table, DADA2 stats table and samples manifest and produces QC plots
- `anospp-plasm` blasts Plasmodium sequences against reference dataset to determine species and infer sample infection status
- `anospp-nn` compares k-mer profiles of mosquito targets against a reference dataset and provides probabilistic species calls
- `anospp-vae` provides finer scale species prediction for An. gambiae complex with VAE projection

## Development

### Setup

Installation is hybrid with conda + poetry:

```bash
git clone git@github.com:malariagen/anospp-analysis.git
cd anospp-analysis
mamba env create -f environment.yml
conda activate anospp_analysis_dev
poetry install
```

### Usage & testing

The code in this repository can be accessed via wrapper scripts:

```bash
anospp-qc \
    --haplotypes test_data/haplotypes.tsv \
    --samples test_data/samples.csv \
    --stats test_data/stats.tsv \
    --outdir test_data/qc
```

Besides, individual components are available as a python API:

```bash
$ python
>>> from anospp_analysis.util import *
>>> PLASM_TARGETS
['P1', 'P2']
```

TODO Automated testing & CI

### Adding Python deps

Introducing python dependencies should be done via poetry:

```bash
poetry add package_name
```

This should update both `pyproject.toml` and `poetry.lock` files

If the package should be used in development environment only, use

```bash
poetry add package_name --dev
```

To update environment after changes made to `pyproject.toml` and/or `poetry.lock`

```bash
poetry install
```

### Adding non-Python deps

Introducing non-python dependencies should be done via conda: edit `environment.yml`,
then re-create the conda environment and poetry deps:

```bash
mamba env create -f environment.yml
conda activate anospp_analysis
poetry install
```

Changes in conda environment might also introduce changes to the python installation,
in which case one should update poetry lock file

```bash
poetry lock
```

## Release checklist

While in dev branch

- test functionality (TODO CI)
- bump version in module init
- bump version in `pyproject.toml`

Then,

- merge into master
- github release
- pypi release

Conda recipe update `conda.recipe/meta.yaml`:

- check deps vs `environment.yaml` and `pyproject.toml`
- bump version in meta yaml  
- update sha256 from pypi project (download files > tar.gz > view hashes)
- test conda recipe with `bioconda-utils build --git-range master` vs bioconda-recipes
