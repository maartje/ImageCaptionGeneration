class Vocabulary:
    def __init__(self):
        self.predefined = []
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addWords(self, words, predefined = False):
        for word in words:
            self.addWord(word, predefined)

    def addWord(self, word, predefined = False):
        if predefined:
            self.predefined.append(word)
        self.word2count[word] = self.word2count.get(word, 0) + 1

    def build_indexes(self, min_occurrence):
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

