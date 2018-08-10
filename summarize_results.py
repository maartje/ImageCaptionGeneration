import os

def summarize_bleu(path):
    path_overview = f'{path}/bleu_summary.txt'
    if os.path.exists(path_overview):
        os.remove(path_overview)
    f_overview = open(path_overview, 'a')
    for mdir in os.listdir(path):
        if not mdir == 'bleu_summary.txt':
            print(mdir, ':', file=f_overview)
            fpath = os.path.join(path, mdir, 'BLEU_val.txt')
            with open(fpath, 'r') as f:
                bleu = f.readlines()[0]
                print('    ', bleu, file = f_overview)

if __name__ == "__main__":
    summarize_bleu('flickr_demo/report')
    summarize_bleu('flickr30k/report')

