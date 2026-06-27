from nltk import word_tokenize
from nltk import CFG
from nltk import ChartParser
from nltk.draw.tree import draw_trees
#import nltk
#nltk.download('punkt')

def read_grammar(grammarfile):
	gf = open(grammarfile)
	return CFG.fromstring(gf.read())

def print_trees(trees):
	for t in trees:
		print(t)

def parse_sentences(grammar, sent):
        parser = ChartParser(grammar)
        tokens = word_tokenize(sent)
        trees = parser.parse(tokens)
        return trees



if __name__ == '__main__':
    sen = 'He worked for the BBC for a decade'
    #sen = 'She spoke to CNN Style about the experience'
    #sen = 'Global warming has caused a change in the pattern of the rainy seasons.'
    #sen = 'I also wonder whether the Davis Cup played a part.'
    #sen = 'The scheme makes money through sponsorship and advertising'
    #sen = 'sentences.txt'
    g = read_grammar("grammar.txt")
    trees = parse_sentences(g,sen)
    for tree in trees:
       draw_trees(tree)
