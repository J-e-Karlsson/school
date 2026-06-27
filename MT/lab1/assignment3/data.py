import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        """ Maps ids to words """
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1 #each word added to the dictionary gets a corresponding id based on its index
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        """Splits the data into sets"""
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.dev = self.tokenize(os.path.join(path, 'dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>'] #End of sequence token to end of each line in file
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word]) #add index aka id to ids
                idss.append(torch.tensor(ids).type(torch.int64)) #turn list of indices (ids) into list of tensors with those values (idss) 
            ids = torch.cat(idss) #merge all tensors in the idss list into one

        return ids