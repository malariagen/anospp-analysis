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

def test_find_normal_edge_simplex():
    a,b,c = np.array([1,0,0]), np.array([4,0,0]), np.array([3,4,0])

    result = vae.find_normal_edge_simplex(
        a,
        b,
        c
    )

    assert (result[0] == np.array([0,1,0])).all()
    assert (result[1] == np.array([1,0,0])).all()

def test_check_half_space():

    n, o = np.array([0,1,0]), np.array([1,0,0])
    points = np.array([[2,-1,11], [4,2,2], [-33,5,-1], [2,0,0]])

    result = np.array([
        vae.check_half_space(p,n,o) for p in points
    ])

    expected_result = np.array([
        [False, True, True, True]
    ])

    assert (result == expected_result).all()
  
def test_compute_distance_to_vertex():

    p = np.array([3,1,2])
    v = np.array([4,0,0])

    result = vae.compute_distance_to_vertex(
        p, 
        v
    )

    assert result == np.sqrt(6)

def test_compute_distance_to_edge():
    v, w = np.array([1,0,0]), np.array([4,0,0])
    points = np.array([[1,2,0], [3,-3,-3], [0,-1,0], [6,1,-1]])

    result = np.array([
        vae.compute_distance_to_edge(p, v, w) for p in points
    ])

    expected_result = np.array([2, np.sqrt(18), 1, np.sqrt(2)])

    assert (result == expected_result).all()

def test_compute_distance_to_plane():
    vertices = np.array([[1,0,0], 
                         [4,0,0], 
                         [3,4,0]])
    points = np.array([[3,1,2], [3,-3,0], [10,1,-1]])

    result = np.array([
        vae.compute_distance_to_plane(p, vertices) for p in points
    ])

    expected_result = np.array([2, 0, 1])

    assert (result == expected_result).all()

def test_check_edge_partition():

    vertices = np.array([[1,0,0], 
                         [4,0,0]])
    points = np.array([[1,4,3], [3,-3,0], [0,-1,-1], [6,1,-1], [3,0,0]])

    result = np.array([
        vae.check_edge_partition(p,
                                vertices) for p in points
    ])

    expected_result = np.array([5, 3, np.sqrt(3), np.sqrt(6), 0])

    assert (result == expected_result).all()

def test_vertex_partition():
    v = np.array([0,1,0])
    vert = np.array([[-10,0,0], [10, 0, 0]])
    points = np.array([[0,2,0], [-1,20,2], [-1,2,0], [-21,3,-1]])

    result = np.array([
        vae.check_vertex_partition(p,
                                 v,
                                 vert) for p in points
    ])

    expected_result = np.array([1, np.sqrt(366), 11/np.sqrt(101), np.sqrt(131)])

    assert (result == expected_result).all()

def test_compute_distance_to_simplex():
    vertices = np.array([[1,0,0], 
                         [4,0,0], 
                         [3,4,0]])
    points = np.array([[3,1,2], [0,-1,2], [3,-3,0], [10,1,-1], [3,1,0], [1,0,0]])

    result = np.array([
        vae.compute_distance_to_simplex(p, vertices) for p in points
    ])

    expected_result = np.array([2, np.sqrt(6), 3, np.sqrt(38), 0, 0])

    assert (result == expected_result).all()

def test_compute_distance_to_hull():
    hull_df = pd.read_csv("ref_databases/gcrefv1/convex_hulls.tsv", sep='\t')
    hull = ConvexHull(hull_df.loc[hull_df.species=='Anopheles_coluzzii', ['mean1', 'mean2', \
                                                                          'mean3']].values)
    pos = hull_df.loc[hull_df.species=='Anopheles_gambiae', ['mean1', 'mean2', 'mean3']].values

    result = vae.compute_distance_to_hull(
        hull, 
        pos
    )

    expected_result = np.array([6.309639,13.038807,6.575978,68.123405,7.7456098,24.679714,\
                        72.25647,76.74071,6.5742974,17.787983,66.25579,73.59986,\
                        76.7035,5.73891,63.732883,23.117966,8.014164,18.58697,\
                        30.661522,31.2423,47.31292,49.995487,49.006126,44.459553,\
                        52.336056,23.796885,59.219532,69.760345,26.823551,4.229266,\
                        77.5418,53.13494,5.8358717,8.693414,2.8645864])
    assert (np.abs(result-expected_result) < 0.0001).all()

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

def test_perform_convex_hull_assignments():
    latent_pos_df = pd.read_csv("test_data/comparisons/latent_coordinates.tsv", sep='\t', \
                             index_col=0)
    vae_samples = np.array(['DN806197N_A1', 'DN806197N_A10', 'DN806197N_A11', 'DN806197N_A12', \
                            'DN806197N_A2', 'DN806197N_A3', 'DN806197N_A4', 'DN806197N_A5', \
                            'DN806197N_A6', 'DN806197N_A7', 'DN806197N_A8', 'DN806197N_A9'])
    latent_pos_df = latent_pos_df.query('sample_id in @vae_samples')
    hull_dict = vae.generate_convex_hulls(pd.read_csv("ref_databases/gcrefv1/convex_hulls.tsv", sep='\t'))

    result = vae.perform_convex_hull_assignments(
        hull_dict, 
        latent_pos_df
    )
    assert (result.VAE_species.values == np.array(['Uncertain_coluzzii_tengrela_gambiae', 'Anopheles_coluzzii', \
                'Anopheles_coluzzii', 'Anopheles_coluzzii', 'Anopheles_coluzzii', 'Uncertain_tengrela_coluzzii_gambiae',\
                'Anopheles_coluzzii', 'Uncertain_coluzzii_tengrela', 'Anopheles_coluzzii', \
                'Anopheles_coluzzii', 'Anopheles_coluzzii', 'Anopheles_coluzzii'])).all()
