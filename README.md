# anospp-analysis
Python package for ANOSPP data analysis

## Usage

TODO
## Development

Clone this repository 
```
git clone git@github.com:malariagen/anospp-analysis.git
```

Install poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

Create development environment
```
cd anospp-analysis
poetry install
```
Activate development environment
```
poetry shell
```
Example wrapper script run
```
python anospp_analysis/qc.py --haplotypes test_data/haplotypes.tsv \
    --samples test_data/samples.csv \
    --stats test_data/stats.tsv \
    --outdir test_data/qc
```

TODO checks & hooks