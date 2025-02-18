import os
import json
import random
import shutil
import argparse
import pandas as pd
import gradio as gr
import pyarabic.number
from glob import glob
import json
import requests
import sys
# running setup file is required for downloading cat and installing the necessary packages
repo_path = os.path.abspath("catt")

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

if __name__ == "__main__":
    
    """
    Usage:
    
    python _code/annotation_interface.py -m /path/to/manifest.csv \
        -r /path/to/audio/files -o /path/to/save/output.csv 
        -s 0 #index to start from if you're resuming annotation
        -d arz_en
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--manifest_file",\
        type=str, help="Path to manifest file",\
        default="/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ArzEn_SpeechCorpus_1.0/test_metadata.csv")
    parser.add_argument("-r","--audio_root", type=str, \
        help="Path to audio files", \
            default="/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ArzEn_SpeechCorpus_1.0/recordings_segmented")
    parser.add_argument("-o","--out_file", type=str, help="Path to save the output", default="./diacritized.csv")
    parser.add_argument("-s","--start_index", type=int, help="Index to start from", default=0)
    parser.add_argument("-d","--dataset", type=str, help="Dataset to use", default="arz_en")

    args = parser.parse_args()
    audio_root = args.audio_root
    manifest_file = args.manifest_file
    out_file = args.out_file
    start = args.start_index
    dataset = args.dataset

    if not os.path.exists(out_file):
        os.system(f"touch {out_file}")

    audio_files = []
    transcripts = []
    
    if dataset == "arz_en":
        print(f"Using {dataset}, with manifest {manifest_file} and audio root {audio_root}. \n Saving output to {out_file}")
        df = pd.read_csv(manifest_file)
        audio_files = df['file_name'].apply(lambda x: os.path.join(audio_root,x)).tolist()
        transcripts = df['transcription'].tolist()
    else:
        # update this for other datasets
        pass
    
    aud_text = list(zip(audio_files, transcripts))[start:]

    index = start

    def next():
        global index
        index += 1
        if index == len(aud_text):
            return gr.Info("No more audio files")
        return aud_text[index][0], aud_text[index][0], aud_text[index][1], gr.Textbox(value=aud_text[index][1]), gr.Textbox(value=f"{index}/{len(aud_text)}", label="Progress")

    def save_new(audio, text):
        if text != "":
            open(out_file, "a").write(f"{audio},{text}\n")
            
    def diacritize(text):         
        text = model.do_tashkeel_batch([text], 1, False)[0]
        return text

    def diacritize_farasa(text):
        url = 'https://farasa.qcri.org/webapi/seq2seq_diacritize/'
        api_key = "ALhwOMJIvjPnrXYHPJ"
        dialect = "mor"
        payload = {'text': text, 'api_key': api_key, "dialect": dialect}
        data = requests.post(url, data=payload)
        if data.ok == True:
            result = json.loads(data.text)
            return result["text"]
        else:
            return "Error"

    def process_diacritize(text, model):
        if model == "Farasa(online)":
            return diacritize_farasa(text)
        if model == "CATT":
            return diacritize(text)

    with gr.Blocks(theme=gr.themes.Glass()) as block:
        progress = gr.Textbox(value=f"{index}/{len(aud_text)}", label="Progress")
        sample = gr.Markdown(f"{aud_text[index][0]}")
        audio = gr.Audio(value=aud_text[index][0], type="filepath", label="Audio", autoplay=True)
        text = gr.Markdown(f"{aud_text[index][1]}")
        selected= gr.Radio(["Farasa(online)", "CATT"], info="Select Model")

        dia_button = gr.Button("Auto Diacritize")
        gr.Markdown('<p style="color:red;">warning: auto diacritization model removes non-arabic characters.</p>')
        correct_text = gr.Textbox(value=f"{aud_text[index][1]}", interactive=True, label="Corrected Text")
        
        with gr.Row():
            save = gr.Button("Save Updated Text")



        dia_button.click(process_diacritize, inputs=[text,selected], outputs=[correct_text])
        save.click(save_new, [audio, correct_text]). \
                then(next, outputs=[sample, audio, text, correct_text])


    
    block.launch(share= True)

