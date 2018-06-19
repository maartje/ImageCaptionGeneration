import os

def read_lines_multiple_files(fpaths):
    for fpath in fpaths:
        for sentence in read_lines(fpath):
            yield sentence
        
    
def read_lines(fpath):
    with open(fpath, 'r') as sentences:
        for sentence in sentences:
            yield sentence
 
