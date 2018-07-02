import os

def check_infiles_exists(fnames):
    file_exists = True
    for fname in fnames:
        if fname and not os.path.isfile(fname):
            print(f"The file '{fname}' does not exists.")
            file_exists = False
    return file_exists

def check_outfiles_exists(fnames):
    file_exists = False
    for fname in fnames:
        if fname and os.path.isfile(fname):
            print(f"A file with the name '{fname}' already exists.")
            file_exists = True
    return file_exists

def ensure_fpath_exists(fname):
    if not fname:
        return 
    dirname = os.path.dirname(fname)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

