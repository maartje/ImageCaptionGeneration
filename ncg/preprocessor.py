from ncg.text_processing.textmapper import TextMapper
import torch
from datetime import datetime
from ncg.debug_helpers import format_duration
import ncg.file_helpers as fh

def preprocess_text_files(fpaths_train, fpaths_val, 
                          fpaths_train_out, fpaths_val_out, fpath_vocab_out,
                          min_occurence = 2):
    fh.ensure_paths_exist(fpaths_train_out + fpaths_val_out + [fpath_vocab_out])
    # TODO: check if out files do not exist unless --overwrite
    
    start_time = datetime.now()
    mapper = build_and_save_vocabulary(fpaths_train, fpath_vocab_out, min_occurence, start_time)
    build_and_save_sentence_vectors(fpaths_train + fpaths_val, 
                                    fpaths_train_out + fpaths_val_out, 
                                    mapper, start_time)

def build_and_save_vocabulary(fpaths_train, fpath_vocab_out, min_occurence, start_time):
    print(f'\n({format_duration(start_time, datetime.now())}) Start building vocabulary ...')
    sentences_train = fh.read_lines_multiple_files(fpaths_train)         
    mapper = TextMapper()
    mapper.build(sentences_train, min_occurence)
    torch.save(mapper, fpath_vocab_out)
    duration_str = format_duration(start_time, datetime.now())
    print(f'({duration_str})    Saved vocabulary file at {fpath_vocab_out}')
    print(f'({format_duration(start_time, datetime.now())}) Finished building vocabulary ...')
    return mapper

def build_and_save_sentence_vectors(fpaths, fpaths_out, mapper, start_time):
    duration_str = format_duration(start_time, datetime.now())
    print(f'\n({duration_str}) Start building indices lists from sentences...')
    for (fpath, fpath_out) in zip(fpaths, fpaths_out):
        sentence_vectors = [mapper.sentence2indices(sentence) for sentence in fh.read_lines(fpath)]
        torch.save(sentence_vectors, fpath_out)
        duration_str = format_duration(start_time, datetime.now())
        print(f'({duration_str})    Saved indices file at {fpath_out}')
    duration_str = format_duration(start_time, datetime.now())
    print(f'({duration_str}) Finished building indices lists from sentences...')
    

    

