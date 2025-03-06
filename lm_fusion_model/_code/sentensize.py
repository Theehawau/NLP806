from pyarabic import araby
from pyarabic.araby import tokenize

dataset = "all_texts"

file=f"data/{dataset}.txt"

with open(file,"r") as f:
    corpus = " ".join(f.readlines())
    
sentences = [sent.strip() for sent in araby.sentence_tokenize(corpus)]

with open(f"data/{dataset}.chunked.pyaraby.txt", "w") as outfile:
    outfile.write("\n".join(sentences))

import nltk
nltk.download('punkt_tab')
from nltk import sent_tokenize

sentences = sent_tokenize(corpus)
with open(f"data/{dataset}.chunked.nltk.txt", "w") as outfile:
    outfile.write("\n".join(sentences))