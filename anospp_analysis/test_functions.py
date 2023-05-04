import vae
import nn
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

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
    vae_samples = np.array(['DN806197N_A1', 'DN806197N_A10', 'DN806197N_A11', 'DN806197N_A12', \
                            'DN806197N_A2', 'DN806197N_A3', 'DN806197N_A4', 'DN806197N_A5', \
                            'DN806197N_A6', 'DN806197N_A7', 'DN806197N_A8', 'DN806197N_A9'])
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
    assert (result.index.values == comparison.index.values).all()
    assert (np.abs(result.mean1.values - comparison.mean1.values) < 0.001).all()
    assert (np.abs(result.mean2.values - comparison.mean2.values) < 0.001).all()
    assert (np.abs(result.mean3.values - comparison.mean3.values) < 0.001).all()

def test_compute_hull_dist():
    hull_df = pd.read_csv("ref_databases/gcrefv1/convex_hulls.tsv", sep='\t')
    hull = ConvexHull(hull_df.loc[hull_df.species=='Anopheles_coluzzii', ['mean1', 'mean2', \
                                                                          'mean3']].values)
    pos = hull_df.loc[hull_df.species=='Anopheles_gambiae', ['mean1', 'mean2', 'mean3']].values

    result = vae.compute_hull_dist(
        hull, 
        pos
    )

    targets = np.array([[6.309639,13.038807,6.575978,68.123405,7.7456098,24.679714,
                        72.25647,76.74071,6.5742974,17.787983,66.25579,73.59986,
                        76.7035,5.73891,63.732883,23.117966,8.014164,18.58697,
                        30.661522,31.2423,47.31292,49.995487,49.006126,44.459553,
                        52.336056,23.796885,59.219532,69.760345,26.823551,4.229266,
                        77.5418,53.13494,5.8358717,8.693414,2.8645864]])
    
    assert (np.abs(result-targets)<0.0001).all()

def test_check_is_in_hull():
    hull_df = pd.read_csv("ref_databases/gcrefv1/convex_hulls.tsv", sep='\t')
    hull_pos = hull_df.loc[hull_df.species=='Anopheles_coluzzii', ['mean1', 'mean2', \
                                                                          'mean3']].values
    positions = np.array([[-84.87537,-4.0789433,-18.389744],
                          [-61.125664,8.855109,-19.726767],
                          [-85, 12, -30],
                          [-74.1326,14.065717,-28.022907],
                          [-52.084938,44.803505,-41.526775],
                          [-6.032356,-62.52318,40.535034]])
    
    result = vae.check_is_in_hull(
        hull_pos, 
        positions)

    assert (result == np.array([False, False, True, True, False, False])).all()

