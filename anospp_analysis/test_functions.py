from vae import select_samples
import pandas as pd

def test_select_samples():
    assignment_df = pd.read_csv('nn/nn_assignment.tsv', sep='\t')
    hap_df = pd.read_csv('nn/non_error_haplotypes.tsv', sep='\t')

    result = select_samples(
        assignment_df,
        hap_df,
        'int'	,
        'Anopheles_gambiae_complex',
        50
    )

    assert len(result[0]) == 723

test_select_samples()
