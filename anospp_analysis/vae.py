import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import argparse
import keras
from scipy.spatial import ConvexHull, Delaunay
from pygel3d import hmesh

from .util import *
from .nn import parse_seqids_series, construct_unique_kmer_table

#Variables
K = 8
LATENTDIM = 3
SEED = 374173
WIDTH = 128
DEPTH = 6
DISTRATIO = 7

def prep_reference_index(reference_version, path_to_refversion):
    '''
    Read in standardised reference index files from database (currently directory)
    '''

    logging.info(f'importing reference index {reference_version}')

    reference_path = f'{path_to_refversion}/{reference_version}/'

    assert os.path.isdir(reference_path), f'reference version {reference_version} does not \
        exist at {reference_path}'

    assert os.path.isfile(f'{reference_path}/selection_criteria.txt'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        selection_criteria.txt file'
    selection_criteria_file = f'{reference_path}/selection_criteria.txt'

    assert os.path.isfile(f'{reference_path}/_weights.hdf5'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        _weights.hdf5 file'
    vae_weights_file = f'{reference_path}/_weights.hdf5'

    assert os.path.isfile(f'{reference_path}/convex_hulls.tsv'), f'reference version \
        {reference_version} at {reference_path} does not contain required \
        convex_hulls.tsv file'
    convex_hulls_df = pd.read_csv(f'{reference_path}/convex_hulls.tsv', sep='\t')

    if os.path.isfile(f'{reference_path}/version.txt'):
        with open(f'{reference_path}/version.txt', 'r') as fn:
            for line in fn:
                version_name = line.strip()
    else:
        logging.warning(f'No version.txt file present for reference version {reference_version} \
                        at {reference_path}')
        version_name = 'unknown'
        
    return selection_criteria_file, vae_weights_file, convex_hulls_df, version_name

def read_selection_criteria(selection_criteria_file, comb_stats_df, hap_df):
    
    level, sgp, n_targets = open(selection_criteria_file).read().split('\t')
    return select_samples(comb_stats_df, hap_df, level, sgp, int(n_targets))

def select_samples(comb_stats_df, hap_df, level, sgp, n_targets):
    '''
    Select the samples meeting the criteria for VAE assignment
    Based on NN assignment and number of targets
    '''
    #identify samples meeting selection criteria
    vae_samples = comb_stats_df.loc[(comb_stats_df[f'res_{level}'] == sgp) & \
                        (comb_stats_df['mosq_targets_recovered'] >= n_targets), 'sample_id']
    #subset haplotype df
    vae_hap_df = hap_df.query('sample_id in @vae_samples')
    
    logging.info(f'selected {len(vae_samples)} samples to be run through VAE')

    return vae_samples, vae_hap_df

def prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids):
    '''
    Prepare k-mer table for a single sample
    '''
    #set up empty arrays
    total_targets = kmers_unique_seqs.shape[0]
    table = np.zeros((total_targets, kmers_unique_seqs.shape[2]), dtype='int')
    n_haps = np.zeros((total_targets), dtype='int')

    for _, row in parsed_seqids.iterrows():
        #only record the first two haplotypes for each target
        if n_haps[row.target] < 2:
            n_haps[row.target] += 1
            table[row.target,:] += kmers_unique_seqs[row.target, row.uidx, :]
    #double counts for homs
    for target in np.arange(total_targets):
        if n_haps[target] == 1:
            table[target,:] *= 2
    #sum over targets
    summed_table = np.sum(table, axis=0)

    return summed_table

def prep_kmers(vae_hap_df, vae_samples, k):
    '''
    Prepare k-mer table for the samples to be run through VAE
    '''
    #translate unique sequences to k-mers
    kmers_unique_seqs = construct_unique_kmer_table(vae_hap_df, k)
    
    logging.info('generating k-mer tables for selected samples')

    #set up k-mer table
    kmers_samples = np.zeros((len(vae_samples), 4**k))

    #fill in table for samples
    for n, sample in enumerate(vae_samples):
        parsed_seqids = parse_seqids_series(vae_hap_df.loc[vae_hap_df.sample_id == sample, 
                                                           'seqid'])
        kmers_samples[n,:] = prep_sample_kmer_table(kmers_unique_seqs, parsed_seqids)

    return(kmers_samples)

def latent_space_sampling(args):
    
    #Add noise to encoder output
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], LATENTDIM),
                                          mean=0, stddev=1., seed=SEED)
    
    return z_mean + keras.backend.exp(z_log_var) * epsilon

def define_vae_input(k):
    input_seq = keras.Input(shape=(4**k,))

    return input_seq

def define_encoder(k):
    input_seq = define_vae_input(k)
    x = keras.layers.Dense(WIDTH, activation = 'elu')(input_seq)
    for i in range(DEPTH-1):
        x = keras.layers.Dense(WIDTH, activation = 'elu')(x)
    z_mean = keras.layers.Dense(LATENTDIM)(x)
    z_log_var = keras.layers.Dense(LATENTDIM)(x)
    z = keras.layers.Lambda(latent_space_sampling, output_shape=(LATENTDIM,), \
                            name = 'z')([z_mean, z_log_var])
    encoder = keras.models.Model(input_seq, [z_mean,z_log_var,z], name = 'encoder')

    return encoder

def define_decoder(k):
    #Check whether you need the layer part here
    decoder_input = keras.layers.Input(shape=(LATENTDIM,), name='ls_sampling')
    x = keras.layers.Dense(WIDTH, activation = "linear")(decoder_input)
    for i in range(DEPTH-1):
        x = keras.layers.Dense(WIDTH, activation = "elu")(x)
    output = keras.layers.Dense(4**k, activation = "softplus")(x)
    decoder=keras.models.Model(decoder_input, output, name = 'decoder')

    return decoder

def define_vae(k):
    input_seq = define_vae_input(k)
    encoder = define_encoder(k)
    decoder = define_decoder(k)
    output_seq = decoder(encoder(input_seq)[2])
    vae = keras.models.Model(input_seq, output_seq, name='vae')

    return vae, encoder

def predict_latent_pos(kmer_table, vae_samples, k, vae_weights_file):

    '''
    Predict latent space of test samples based on reference database
    '''

    vae, encoder = define_vae(k)

    vae.load_weights(vae_weights_file)
    predicted_latent_pos = encoder.predict(kmer_table)

    predicted_latent_pos_df = pd.DataFrame(index=vae_samples, columns=['mean1', 'mean2', 'mean3',\
                                                                       'sd1', 'sd2', 'sd3'])
    for i in range(3):
        predicted_latent_pos_df[f'mean{i+1}'] = predicted_latent_pos[0][:,i]
        predicted_latent_pos_df[f'sd{i+1}'] = predicted_latent_pos[1][:,i]

    return predicted_latent_pos_df

def generate_convex_hulls(convex_hulls_df):
    '''
    Read in pre-computed points for convex hulls for each species
    '''
    logging.info('setting up convex hulls from reference database')
    hull_dict = dict()
    for species in convex_hulls_df.species.unique():
        pos = convex_hulls_df.loc[convex_hulls_df.species==species, ['mean1', 'mean2', \
                                                                     'mean3']].values
        hull = ConvexHull(pos)
        hull_dict[species] = (pos, hull)

    return hull_dict

def compute_hull_dist(hull, positions):
    '''
    Compute the distance to the specified hull for a set of positions
    '''
    manifold = hmesh.Manifold().from_triangles(hull.points, hull.simplices)   
    mesh_dist = hmesh.MeshDistance(manifold)
    dist = mesh_dist.signed_distance(positions.flatten())

    return np.absolute(dist)

def check_is_in_hull(hull_pos, positions):
    '''
    Check whether a set of positions lies inside the specified hull
    '''
    if not isinstance(hull_pos,Delaunay):
        hull = Delaunay(hull_pos)

    in_hull = hull.find_simplex(positions)>=0

    return in_hull

def get_unassigned_samples(latent_positions_df):
    unassigned = latent_positions_df.loc[latent_positions_df.VAE_species.isnull()].index
    n_unassigned = len(unassigned)
    return unassigned, n_unassigned

def generate_hull_dist_df(hull_dict, latent_positions_df, unassigned):

    dist_df = pd.DataFrame(index=unassigned)
    positions = latent_positions_df.loc[unassigned,['mean1','mean2','mean3']].values
    for species in hull_dict.keys():
        dist_df[species] = compute_hull_dist(hull_dict[species][1], positions)
    return dist_df 

def get_closest_hulls(hull_dict, latent_positions_df, unassigned):

    dist_df = generate_hull_dist_df(hull_dict, latent_positions_df, unassigned)
    summary_dist_df = pd.DataFrame(index=dist_df.index)
    summary_dist_df['dist1'] = dist_df.min(axis=1)
    summary_dist_df['species1'] = dist_df.idxmin(axis=1)
    summary_dist_df['dist2'] = dist_df.apply(lambda x: x.sort_values()[1], axis=1)
    summary_dist_df['species2'] = dist_df.apply(lambda x: x.sort_values().index[1], axis=1)
    return summary_dist_df, dist_df

def assign_gam_col_band(latent_positions_df, summary_dist_df):
    #Determine which samples are in gamcol band
    gamcol_band = summary_dist_df.loc[(summary_dist_df.species1.isin(['Anopheles_gambiae', \
                            'Anopheles_coluzzii'])) & (summary_dist_df.species2.isin([\
                            'Anopheles_gambiae', 'Anopheles_coluzzii'])) & \
                            (summary_dist_df.dist2<14)].copy()
    #Make assignments for the samples in gamcol band
    if gamcol_band.shape[0]>0:
        gamcol_dict = dict(gamcol_band.apply(lambda row: 'Uncertain_'+row.species1.split('_')[1]+\
                                             '_'+row.species2.split('_')[1], axis=1))
        print(gamcol_dict) 
        latent_positions_df.loc[gamcol_band.index, 'VAE_species'] = latent_positions_df.loc[\
            gamcol_band.index].index.map(gamcol_dict)
    return latent_positions_df, gamcol_band.shape[0]

def assign_to_closest_hull(latent_positions_df, summary_dist_df, unassigned):
    '''
    Assign samples that are much closer to one hull than to all others to that hull
    Currently defined as a distance 7 times smaller than all others
    '''
    fuzzy_hulls = summary_dist_df.loc[(DISTRATIO*summary_dist_df.dist1 < \
        summary_dist_df.dist2)&(summary_dist_df.index.isin(unassigned))].copy()
    if fuzzy_hulls.shape[0]>0:
        fuzzy_hulls_dict = dict(fuzzy_hulls.species1)
        latent_positions_df.loc[unassigned, 'VAE_species'] = latent_positions_df.loc[\
            unassigned].index.map(fuzzy_hulls_dict)
    return latent_positions_df, fuzzy_hulls.shape[0]

def assign_to_multiple_hulls(latent_positions_df, dist_df, unassigned):
    '''
    Assign samples in between hulls to multiple hulls in order of closeness
    '''
    between_hulls = dist_df.loc[dist_df.index.isin(unassigned)].copy()
    between_hulls_dict = dict(between_hulls.apply(lambda row: 'Uncertain_'+'_'.join(\
        label.split('_')[1] for label in row.sort_values().index[row.sort_values()<\
                                                            DISTRATIO*row.min()]), axis=1))
    latent_positions_df.loc[unassigned, 'VAE_species'] = latent_positions_df.loc[\
        unassigned].index.map(between_hulls_dict)
    return latent_positions_df, between_hulls.shape[0]

def perform_convex_hull_assignments(hull_dict, latent_positions_df):
    '''
    Perform convex hull assignments based on latent space positions
    '''
    logging.info('performing convex hull assignment')
    positions = latent_positions_df[['mean1', 'mean2', 'mean3']].values
    
    #first check which samples fall inside convex hulls
    for label in hull_dict.keys():
        is_in_hull = check_is_in_hull(hull_dict[label][0], positions)
        latent_positions_df.loc[is_in_hull, 'VAE_species'] = label
    
    #Record unassigned samples 
    unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
    logging.info(f'{latent_positions_df.shape[0] - n_unassigned} samples fall inside convex hulls; \
{n_unassigned} samples still to be assigned')
    
    #for the unassigned samples, get distances to two closest hulls
    if n_unassigned > 0:
        summary_dist_df, dist_df = get_closest_hulls(hull_dict, latent_positions_df, unassigned)
        latent_positions_df, n_newly_assigned = assign_gam_col_band(latent_positions_df, \
                                                                    summary_dist_df)
        unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
        logging.info(f'{n_newly_assigned} samples assigned to uncertain_gambiae_coluzzii or \
uncertain_coluzzii_gambiae; {n_unassigned} samples still to be assigned')
    if n_unassigned > 0:
        latent_positions_df, n_newly_assigned = assign_to_closest_hull(latent_positions_df, \
                                                                summary_dist_df, unassigned)
        unassigned, n_unassigned = get_unassigned_samples(latent_positions_df)
        logging.info(f'{n_newly_assigned} additional samples assigned to a single species; \
{n_unassigned} samples still to be assigned')
    if n_unassigned > 0:
        latent_positions_df, n_newly_assigned = assign_to_multiple_hulls(latent_positions_df,\
                                                                        dist_df, unassigned)
        logging.info(f'{n_newly_assigned} samples assigned to multiple hulls.')
    logging.info(f'finished assigning {latent_positions_df.shape[0]} samples')

    return latent_positions_df


def vae(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP VAE data import started')

    hap_df = pd.read_csv(args.haplotypes, sep='\t')
    comb_stats_df = pd.read_csv(args.manifest, sep='\t')

    selection_criteria_file, vae_weights_file, convex_hulls_df, version_name = prep_reference_index(\
        args.reference_version, args.path_to_refversion)
    vae_samples, vae_hap_df = read_selection_criteria(selection_criteria_file,\
                                 comb_stats_df, hap_df)
    kmer_table = prep_kmers(vae_hap_df, vae_samples, K)

    latent_positions_df = predict_latent_pos(kmer_table, vae_samples, K, vae_weights_file)
    latent_positions_df.to_csv(f'{args.outdir}/latent_positions.tsv', sep='\t')

    hull_dict = generate_convex_hulls(convex_hulls_df)

    ch_assignment_df = perform_convex_hull_assignments(hull_dict, latent_positions_df)





    

    logging.info('All done!')

    
def main():
    
    parser = argparse.ArgumentParser("VAE assignment for samples in gambiae complex")
    parser.add_argument('-a', '--haplotypes', help='Haplotypes tsv file with errors removed \
                        as generated by anospp-nn', required=True)
    parser.add_argument('-m', '--manifest', help='Sample assignment tsv file as generated by\
                        anospp-nn', required=True)
    parser.add_argument('-r', '--reference_version', help='Reference index version - \
                        currently a directory name. Default: gcrefv1', default='gcrefv1')
    parser.add_argument('-o', '--outdir', help='Output directory. Default: nn', default='vae')
    parser.add_argument('--path_to_refversion', help='path to reference index version.\
         Default: ref_databases', default='ref_databases')
    parser.add_argument('--no_plotting', help='Do not generate plots. Default: False', \
                        default=False)
    parser.add_argument('-v', '--verbose', 
                        help='Include INFO level log messages', action='store_true')

    args = parser.parse_args()
    args.outdir=args.outdir.rstrip('/')
    vae(args)

if __name__ == '__main__':
    main()