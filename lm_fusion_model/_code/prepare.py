
import re

import pyarabic.number
an = pyarabic.number.ArNumbers()
an.int2str('125')
from pyarabic.number import vocalize_number

dataset = "all_texts.chunked.pyaraby"

def contains_english_or_numbers(sentence):
    # Regular expression pattern to match English letters and punctuation
    pattern = re.compile(r'[A-Za-z0-9]')
    # Search the pattern in the sentence
    return not bool(pattern.search(sentence))

def is_number(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    return bool(pattern.match(s))

def replace_arabic_punctuation_with_space(text):
    # List of Arabic punctuation marks 
    arabic_punctuation = "،؛.؟,\/#!$%\^&\*«;:»=‫‏‮‬‎\-'_`»!~()\{\}\[\]\""
    
    # Create a pattern to match any of the Arabic punctuation marks
    pattern = re.compile(f"[{re.escape(arabic_punctuation)}]")
    
    # Replace matched punctuation marks with space
    cleaned_text = pattern.sub("", text)
    
    # Return the cleaned text
    return cleaned_text


# NO numbers approach 

# file="data/{dataset}.chunked.no_numbers.txt"
# # 1. Read the sentences from the file and strip the newline character
# with open(file,"r") as f:
#     sentences = [s.replace("\n","").strip() for s in f.readlines()]
    
# lengths = [len(sent) for sent in sentences]   
# # Statistics about the sentences
# print("Statistics before chunking...")
# print("Number of sentences: ", len(sentences))
# print("Average sentence length: ", sum(lengths) / len(sentences))
# print("Max sentence length: ", max(lengths))
# print("Min sentence length: ", min(lengths))
# print("Number of sentences longer than 500 characters: ", len([sent for sent in sentences if len(sent) > 500]))
# print("Number of sentences shorter than 20 characters: ", len([sent for sent in sentences if len(sent) < 20]))

# # 2. Chunk by comma if length is greater than 500
# chunked_sentences = []
# for sent in sentences:
#     if len(sent) > 500:
#         chunked_sentences.extend(sent.split("،"))
#     else:
#         chunked_sentences.append(sent)
        
#     # , exclude sentences shorter than 20 characters
# chunked_sentences = [sent for sent in chunked_sentences if len(sent) > 20]

# print("Statistics after chunking...")
# print("Number of sentences: ", len(chunked_sentences))
# print("Average sentence length: ", sum([len(sent) for sent in chunked_sentences]) / len(chunked_sentences))
# print("Max sentence length: ", max([len(sent) for sent in chunked_sentences]))
# print("Min sentence length: ", min([len(sent) for sent in chunked_sentences]))
    
# # 3. Normalize the Arabic text by replacing Arabic punctuation marks with space
# normalized_sentences = [replace_arabic_punctuation_with_space(sent).strip() for sent in chunked_sentences]

# with open("data/{dataset}.cleaned.no_numbers.txt", "w") as outfile:
#     outfile.write("\n".join(normalized_sentences))

# def print_statistics(chunked_sentences):
#     print("Number of sentences: ", len(chunked_sentences))
#     print("Average sentence length: ", sum([len(sent) for sent in chunked_sentences]) / len(chunked_sentences))
#     print("Max sentence length: ", max([len(sent) for sent in chunked_sentences]))
#     print("Min sentence length: ", min([len(sent) for sent in chunked_sentences]))



class Prepare():
    def __init__(self, file,out_file, type=None):
        self.file = file
        self.out_file = out_file
        self.sentences = []
        self.chunked_sentences = []
        self.normalized_sentences = []
        self.type = type
    
    def read_sentences(self):
        with open(self.file,"r") as f:
            self.sentences = [s.replace("\n","").strip() for s in f.readlines()]
    
    def chunk_sentences(self):
        self.chunked_sentences = []
        for sent in self.sentences:
            if len(sent) > 450:
                self.chunked_sentences.extend(sent.split("،"))
            else:
                self.chunked_sentences.append(sent)
        
        self.chunked_sentences = [sent for sent in self.chunked_sentences if len(sent) > 20]
    
    def normalize_sentences(self):
        self.normalized_sentences = [replace_arabic_punctuation_with_space(sent).strip() for sent in self.chunked_sentences]
        
    def write_sentences(self):
        with open(self.out_file, "w") as outfile:
            outfile.write("\n".join(self.normalized_sentences))
            
    def convert_numbers(self):
        s = []
        for sent in self.sentences:
            sent_as_list = sent.split(" ")
            for i, word in enumerate(sent_as_list):
                if is_number(word):
                    sent_as_list[i] = "".join(vocalize_number(an.int2str(word)))
            s.append(" ".join(sent_as_list))
        self.sentences = s
    
    @staticmethod
    def print_statistics(chunked_sentences):
        print("Number of sentences: ", len(chunked_sentences))
        print("Average sentence length: ", sum([len(sent) for sent in chunked_sentences]) / len(chunked_sentences))
        print("Max sentence length: ", max([len(sent) for sent in chunked_sentences]))
        print("Min sentence length: ", min([len(sent) for sent in chunked_sentences]))
          
    def run(self):
        self.read_sentences()
        print("Statistics before chunking...")
        self.print_statistics(self.sentences)
            
        if self.type == "has_numbers_convert":  
            print("Converting numbers to Arabic text...") 
            self.convert_numbers()
            print("Statistics after converting numbers")
            self.print_statistics(self.sentences)
        self.chunk_sentences()
        print("Statistics after chunking...")
        self.print_statistics(self.chunked_sentences)
        self.normalize_sentences()
        self.write_sentences()
        

# prep = Prepare(f"data/{dataset}.chunked.no_numbers.txt", f"data/{dataset}.cleaned.no_numbers.txt")
# prep.run()        
        
prep = Prepare(f"data/{dataset}.txt", f"data/{dataset}.chunked.txt")
prep.run()

# prep = Prepare(f"data/{dataset}.chunked.has_numbers.txt", f"data/{dataset}.converted.has_numbers.txt", type="has_numbers_convert")
# prep.run()
