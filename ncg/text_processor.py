from ncg.vocabulary import Vocabulary
from mosestokenizer import MosesTokenizer, MosesDetokenizer

EOS = "EOS"
SOS = "SOS"
UNKNOWN = "UNKNOWN"

def build_vocabulary(sentences, min_occurence):
    vocab = Vocabulary()
    vocab.addWords([SOS, EOS, UNKNOWN], predefined = True)
    for sentence in sentences:
        tokens = sentence2tokens(sentence)
        vocab.addWords(tokens)
    vocab.build_indexes(min_occurence)
    return vocab

def sentence2sentence(sentence): 
    return tokens2sentence(sentence2tokens(sentence))

def sentence2indices(sentence, vocab):
    return tokens2indices(sentence2tokens(sentence), vocab)

def indices2sentence(indices, vocab):
    return tokens2sentence(indices2tokens(indices, vocab))

def sentence2tokens(sentence):
    sentence_lc = sentence.lower().strip()
    with MosesTokenizer('en') as tokenize:
        tokens = tokenize(sentence_lc)
    tokens.insert(0, "SOS")
    tokens.append("EOS")
    return tokens

def tokens2sentence(tokens):
    with MosesDetokenizer('en') as detokenize:
        sentence = detokenize(tokens[1:-1]) # remove SOS, EOS tokens
    # REMARK: We do not use true casing, instead we lowercase reference sentences
    return sentence

def tokens2indices(tokens, vocab):
    def token2index(t):
        if vocab.word2index.get(t):
            return vocab.word2index[t] 
        else:
            return vocab.word2index[UNKNOWN]
    return [token2index(t) for t in tokens]

def indices2tokens(indices, vocab):
    return [vocab.index2word[i] for i in indices]    
