import os
import re
import sys
import torch
import librosa
import evaluate
import argparse
import unicodedata
import pandas as pd
import numpy as np

from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

datasets_to_evaluate = {
    "clartts": "/l/speech_lab/data_806/CLARTTS",
    "asc": "herwoww/asc",
    "mdpc": "herwoww/mdpc",
}

punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()

def predict(example, use_hf=True):
    if use_hf:
        if example["audio"]['sampling_rate'] != 16000:
            audio = librosa.resample(np.array(example["audio"]['array']), orig_sr=example["audio"]['sampling_rate'], target_sr=16000)
        else:
            audio = np.array(example["audio"]['array'])
        result = pipe(audio)
        example['pred'] = remove_punctuation(result["text"]) # remove punctuation from prediction
        example['processed_text'] = remove_punctuation(example["transcription"]) # remove diacritics and punctuation from ground truth
        return example
    else: 
        # example is path to wav file
        wav, sr = librosa.load(example, sr=16000)
        result = pipe(wav)
        pred = remove_punctuation(result["text"])
        return pred

wer = evaluate.load("wer")
cer = evaluate.load("cer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="whisper-tiny-clartts")
    parser.add_argument("--dataset", type=str, default="clartts,asc,mdpc")
    parser.add_argument("--use_csv", action="store_true")
    parser.add_argument("--csv_file", type=str, default="predictions.csv")
    args = parser.parse_args()
    
    model_id = args.model_name
    save_dir = model_id.replace("/best", "")
    
    if args.use_csv:
        print("Using existing predictions file")
        df = pd.read_csv(args.csv_file, sep="\t")
        r = df['references'].apply(lambda x: remove_punctuation(x))
        p = df['predictions'].apply(lambda x: remove_punctuation(x))
        _wer = 100 * wer.compute(predictions=p, references=r)
        _cer = 100 * cer.compute(predictions=p, references=r)
        print(f"CSV file: {args.csv_file}")
        print("Model: ", model_id)
        print("WER: ", _wer)
        print("CER: ", _cer)

    else:
        print("Running Generation and Evaluation")
        print("Loading model and pipeline")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        print("Running evaluation...")
        for dataset in args.dataset.split(","):
            if dataset not in datasets_to_evaluate:
                raise ValueError(f"Dataset {dataset} not found in the list of datasets to evaluate")
            print("Loading dataset...")
            df = load_dataset(datasets_to_evaluate[dataset], split="test")
            # if dataset == "clartts":
            #     df = df.map(lambda x: predict(x, use_hf=False))
            # else:
            df = df.map(lambda x: predict(x, use_hf=True))
            r = df['processed_text']
            p = df['pred']

            _wer = 100 * wer.compute(predictions=p, references=r)
            _cer = 100 * cer.compute(predictions=p, references=r)
            print(f"Dataset: {dataset}")
            print("Model: ", model_id)
            print("WER: ", _wer)
            print("CER: ", _cer)
            
            print("Saving predictions to CSV")
            out_file = open(f"{save_dir}/predictions_{dataset}.csv", "w")
            
            print("predictions\treferences", file=out_file)
            for i, item in enumerate(p):
                print(f"{p[i]}\t{r[i]}", file=out_file)

            out_file.close()
    
            