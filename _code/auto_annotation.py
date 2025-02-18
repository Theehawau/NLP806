import argparse
from time import sleep

# Auto annotation  for the dataset
# step 1 read row data (id,filename, transcript and duration)
# find latin words and remove them from the sentence, and store them in dictionary with their locations in the sentence
# annotate the arabic sentence using farasa seq2seq diaractirize
# restore the latin words to the sentence
# save the new transcript in new column

import pandas as pd
import re
import json
import requests
import sys, os

from tqdm import tqdm

# running setup file is required for downloading cat and installing the necessary packages
repo_path = os.path.abspath("../catt")

sys.path.append(repo_path)
if repo_path not in sys.path:
    sys.path.append(repo_path)
import torch

from ed_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer

#from utils import remove_non_arabic
tokenizer = TashkeelTokenizer()
# update this path , see the repo for model download link
ckpt_path = repo_path + '/models/best_ed_mlm_ns_epoch_178.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_seq_len = 1024
model = TashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=3, learnable_pos_emb=False)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval().to(device)
# Sample data for testing only
data = {
    "ID": ["52_1_0", "27_274_0", "49_83_2", "26_53_2", "73_70_1", "Melekher10_243_2", "50_256_0", "662_148_0", "23_52"],
    "file_name": [
        "52_1_0.wav", "27_274_0.wav", "49_83_2.wav", "26_53_2.wav", "73_70_1.wav",
        "Melekher10_243_2.wav", "50_256_0.wav", "662_148_0.wav", "23_52.wav"
    ],
    "transcription": [
        "الحركة النسوية كيفاش بدات و علاش بدات و شنوا المطالب متع النسويات ضيفتي ليوم رانية عطافي feminist a book reviewer an english teacher و لfounder متع طبرقة book club و a point كتبت",
        "vision و لحقيقة أنا مالعباد لي ما يسهروش برشا surtout بش يقراو معنتها نجم نسهر على فيلم it's ok أما بش نقرا ما نجمش تجيني نرقد يعني",
        "نجم نجم نبدا معناها ب des gestes des gestesصغار",
        "﻿كيفاش جاتك الفكرةok donc",
        "ديما you're low-key عرفت ديما low-key",
        "donc كل شي لازمو يكون خارج منك إنتي و من حاجة بالحق إنتي تحبها",
        "في أنك تكون أسود ولا سوداء قبل لا معنتها ساعة أولا حتى حد ما عندو accés le monde exterieur كونشي سافرت و كونشي معنتها صارت هجرة",
        "question كان تحبوا على relationship advice just text me ولا ما نعرفش عليكم i will try to answer every معناها",
        "تي عصابة مش شلة"
    ],
    "duration": [12.0, 9.929, 3.267, 5.493, 4.019, 5.643, 11.366, 12.0, 1.214]
}

#df = pd.DataFrame(data)

def extract_latin_words(sentence):
    words = sentence.split()
    latin_words = {}
    for i, word in enumerate(words):
        if re.search(r'[a-zA-Z]', word):  # Check if word contains Latin characters
            latin_words[i] = word
    return latin_words

def remove_latin_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if not re.search(r'[a-zA-Z]', word)]
    return " ".join(filtered_words)

def restore_latin_words(sentence, latin_words):
    words = sentence.split()
    for index, word in latin_words.items():
        words.insert(index, word)
    return " ".join(words)

def diacritic_catt(text):
    text = model.do_tashkeel_batch([text], 1, False)[0]
    return text

def diacritize_farasa(text):
        url = 'https://farasa.qcri.org/webapi/seq2seq_diacritize/'
        api_key ="MwJTHPmcClywMrZmXV" # "ALhwOMJIvjPnrXYHPJ"
        dialect = "mor"
        payload = {'text': text, 'api_key': api_key, "dialect": dialect}
        data = requests.post(url, data=payload)
        if data.ok == True:
            result = json.loads(data.text)
            return True, result["text"]
        else:
            return False, "Error"
# Processing
def process(inputfile,outputfile,modelname,verbose):
    # read ing dataframe
    df=pd.read_csv(inputfile)

    latin_words_dicts = []
    cleaned_transcriptions = []
    annotated_transcriptions = []

    for transcript in tqdm(df["transcription"]):
        latin_words = extract_latin_words(transcript)
        latin_words_dicts.append(latin_words)
        cleaned_sentence = remove_latin_words(transcript)
        # Diacritize
        if modelname =="farasa":
            check,annotated_sentence=diacritize_farasa(cleaned_sentence)
            print(annotated_sentence)
            sleep(1)
        if modelname == "catt":
            annotated_sentence= diacritic_catt(cleaned_sentence)
            if(verbose == 1):
                print(annotated_sentence)
        restored_sentence = restore_latin_words(annotated_sentence, latin_words)
        cleaned_transcriptions.append(cleaned_sentence)
        annotated_transcriptions.append(restored_sentence)

    df["latin_words"] = latin_words_dicts
    df["cleaned_transcription"] = cleaned_transcriptions
    df["annotated_transcription"] = annotated_transcriptions

    df.to_csv(outputfile,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--manifest_file", type=str, help="Path to manifest file")
    parser.add_argument("-o", "--out_file", type=str, help="Path to save the output", default="./diacritized.csv")
    parser.add_argument("-m", "--module", type=str, help="catt|farasa", default="catt")
    parser.add_argument("-v", "--verbose", type=int, help="0 or 1 for debuging", default=0)

    args = parser.parse_args()
    manifest_file = args.manifest_file
    out_file = args.out_file
    module_type = args.module
    verbose = args.verbose
    process(manifest_file, out_file, module_type,verbose)

