from os import system

def calculate_bleu(fpaths_references, fpath_predicted, fpath_out):
    fpaths_references_str = ' '.join(fpaths_references)
    system(
        f'./ncg/scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_predicted} > {fpath_out}'
    )

