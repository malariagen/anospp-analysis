import vae
import nn
import pandas as pd
import numpy as np

def test_select_samples():
    assignment_df = pd.read_csv('test_data/output/nn_assignment.tsv', sep='\t')
    hap_df = pd.read_csv('test_data/output/non_error_haplotypes.tsv', sep='\t')

    result = vae.select_samples(
        assignment_df,
        hap_df,
        'int'	,
        'Anopheles_gambiae_complex',
        50
    )

    assert len(result[0]) == 723
    assert (result[1]).shape == (63138, 9)

def test_prep_sample_kmer_table():
    hap_df = pd.read_csv('test_data/output/non_error_haplotypes.tsv', sep='\t')
    kmers_unique_seqs = nn.construct_unique_kmer_table(hap_df, 8)
    parsed_seqids = nn.parse_seqids_series(hap_df.loc[hap_df.sample_id=='DN806197N_A1',\
                                                      'seqid'])

    result = vae.prep_sample_kmer_table(
        kmers_unique_seqs, 
        parsed_seqids
    )

    assert result.shape == (65536,)
    assert result.sum() == 18374

def test_prep_kmers():
    hap_df = pd.read_csv('test_data/output/non_error_haplotypes.tsv', sep='\t')
    vae_samples = np.array(['DN806197N_A1', 'DN806197N_A2', 'DN806197N_A3', 'DN806197N_A4', \
                            'DN806197N_A5', 'DN806197N_A6', 'DN806197N_A7', 'DN806197N_A8', \
                            'DN806197N_A9', 'DN806197N_A10', 'DN806197N_A11', 'DN806197N_A12'])
    vae_hap_df = hap_df.query('sample_id in @vae_samples')

    result = vae.prep_kmers(
        vae_hap_df,
        vae_samples,
        8
    )

    assert result.shape == (12, 65536)
    assert (result.sum(axis=1) == np.array([18374, 18628, 18357, 18634, 18364, 18326, 18613, \
                                            18627, 18629, 18619, 18655, 18354])).all()

def test_predict_latent_pos():
    hap_df = pd.read_csv('test_data/output/non_error_haplotypes.tsv', sep='\t')
    vae_samples = np.array(['DN806197N_A1', 'DN806197N_A2', 'DN806197N_A3', 'DN806197N_A4', \
                            'DN806197N_A5', 'DN806197N_A6', 'DN806197N_A7', 'DN806197N_A8', \
                            'DN806197N_A9', 'DN806197N_A10', 'DN806197N_A11', 'DN806197N_A12'])
    vae_hap_df = hap_df.query('sample_id in @vae_samples')
    kmer_table = vae.prep_kmers(vae_hap_df, vae_samples, 8)
    comparison = pd.read_csv("test_data/comparisons/latent_coordinates.tsv", sep='\t', \
                             index_col=0)

    result = vae.predict_latent_pos(
        kmer_table, 
        vae_samples, 
        8,
        'ref_databases/gcrefv1/_weights.hdf5'
    )

    assert (result.mean1 == comparison.mean1).all()
    assert (result.mean2 == comparison.mean2).all()
    assert (result.mean3 == comparison.mean3).all()
