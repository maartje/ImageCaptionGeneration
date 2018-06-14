class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addWords(self, words, predefined = False):
        for word in words:
            self.addWord(word, predefined)

    def addWord(self, word, predefined = False):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        if not predefined:
            self.word2count[word] = self.word2count.get(word, 0) + 1

    def removeLowFrequentWords(self, min_occurrence):
        low_frequency_words = [
            word for word, count in vocab.word2count.items() if count < min_occurrence] 
        for word in low_frequency_words:
            self._removeWord(word)
    
    def _removeWord(self, word):
        self.word2count.pop(word, None)
        index = self.word2index.pop(word, None)
        self.index2word.pop(index, None)
        self.n_words -= 1

