from mosestokenizer import MosesTokenizer, MosesDetokenizer

def sentence2sentence(sentence): 
    return tokens2sentence(sentence2tokens(sentence))

def sentence2tokens(sentence):
    sentence_lc = sentence.lower().strip()
    with MosesTokenizer('en') as tokenize:
        tokens = tokenize(sentence_lc)
    return tokens

def tokens2sentence(tokens):
    with MosesDetokenizer('en') as detokenize:
        sentence = detokenize(tokens) 
    # REMARK: We do not use true casing, instead we lowercase reference sentences
    return sentence


