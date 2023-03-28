# anospp-analysis
Python package for ANOSPP data analysis

## Installation

TODO

## Documentation

TODO

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
python anospp_analysis/qc.py \
    --haplotypes test_data/haplotypes.tsv \
    --samples test_data/samples.csv \
    --stats test_data/stats.tsv \
    --outdir test_data/qc
```

Besides, individual functions are available as an API 

TODO testing & CI

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
