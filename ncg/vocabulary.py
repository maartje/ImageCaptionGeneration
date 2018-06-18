from ncg.text_processor import sentence2tokens

class Vocabulary:
    def __init__(self):
        self.predefined = []
        self.word2count = {}
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def build(self, sentences, predefined = [], min_occurence = 1):
        self.predefined = predefined
        self._count_word_occurrences(sentences)
        self._build_indexes(min_occurence)

    def _count_word_occurrences(self, sentences):
        self.word2count = {}
        for sentence in sentences:
            words = sentence2tokens(sentence)
            for word in words:
                self.word2count[word] = self.word2count.get(word, 0) + 1
    
    def _build_indexes(self, min_occurrence):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0
        for word in self.predefined:
            self._addWordToIndex(word)
        for word, count in self.word2count.items():
            if (count >= min_occurrence): 
                self._addWordToIndex(word)
                        
    def _addWordToIndex(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


