import nltk

from nltk.corpus import gutenberg 
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk import WordNetLemmatizer #ta bort?


""" Behövs dessa? """
import string
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import Counter
import re

#Ta bort oanvända
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def process_data():
    


