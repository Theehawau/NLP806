import os
import re
import sys
import torch
import librosa
import evaluate
import unicodedata
import pandas as pd


df = pd.read_csv("/l/users/hawau.toyin/NLP806/asr/models/whisper-tiny-clartts/predictions.csv", sep="\t")

punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()

wer = evaluate.load("wer")
cer = evaluate.load("cer")

r = df['references'].apply(lambda x: remove_punctuation(x))
p = df['predictions'].apply(lambda x: remove_punctuation(x))

_wer = 100 * wer.compute(predictions=p, references=r)
_cer = 100 * cer.compute(predictions=p, references=r)

print("WER: ", _wer)
print("CER: ", _cer)