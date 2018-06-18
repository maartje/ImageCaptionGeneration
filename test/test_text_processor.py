"""
Here come the tests for the text processor that tokenizes and preprocesses the text
"""

import unittest
from ncg.text_processor import preprocess, sentence2tokens, tokens2sentence

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.sentence = 'Hello world, hello Foo!  \n'

    def test_preprocess(self):
        sentence_expected = 'hello world, hello foo!'
        self.assertEqual(sentence_expected, preprocess(self.sentence))

    def test_sentence2tokens(self):
        self.assertEqual(['hello', 'world', ',', 'hello', 'foo', '!'], sentence2tokens(self.sentence))

    def test_tokens2sentence(self):
        sentence_expected = 'hello world, hello foo!'
        sentence_actual = tokens2sentence(sentence2tokens(self.sentence))
        self.assertEqual(sentence_expected, sentence_actual)

if __name__ == '__main__':
    unittest.main()


