import os

def check_files_exist(fnames):
    file_exists = True
    for fname in fnames:
        if not os.path.isfile(fname):
            print(f"The file '{fname}' does not exists.")
            file_exists = False
    if not file_exists:
        raise FileNotFoundError('One or more input files do not exist.')

def check_files_not_exist(fnames):
    file_exists = False
    for fname in fnames:
        if os.path.isfile(fname):
            print(f"A file with the name '{fname}' already exists.")
            file_exists = True
    if file_exists:
        raise FileExistsError('One or more output files already exist')

def ensure_paths_exist(fnames):
    dirnames = {os.path.dirname(fname) for fname in fnames}
    for dirname in dirnames:
        ensure_dir_exists(dirname)
                
def ensure_dir_exists(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

