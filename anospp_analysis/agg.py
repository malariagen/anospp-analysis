import pandas as pd
import numpy as np
import os
import argparse

def agg(args):

    setup_logging(verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok =True)

    logging.info('ANOSPP results merging started')

    
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