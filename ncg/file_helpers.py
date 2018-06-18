import os

def read_lines_multiple_files(fpaths):
    for fpath in fpaths:
        for sentence in read_lines(fpath):
            yield sentence
        
    
def read_lines(fpath):
    with open(fpath, 'r') as sentences:
        for sentence in sentences:
            yield sentence

def ensure_paths_exist(paths):
    for path in paths:
        ensure_path_exists(path)
  
def ensure_path_exists(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

