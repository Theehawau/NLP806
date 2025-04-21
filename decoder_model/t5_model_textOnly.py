import re
import os
import torch
import librosa
import evaluate
import warnings
import unicodedata

import pandas as pd
import torch.nn as nn
import soundfile as sf

from tqdm import tqdm
from datasets import load_dataset, Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5ForSpeechToText

warnings.filterwarnings("ignore")

dataset = "ClArTTS+TunSwitch"
# PATH=f'_models/speecht5+lstm_model_lr_0.001_best_{dataset}.pth'
PATH=f'_models/textonly_speecht5+lstm_model_lr_0.001_best_{dataset}.pth'


# dataset="ClArTTS"
# PATH=f'_models/speecht5+lstm_model_lr_0.001_best.pth'

cer = evaluate.load("cer")
wer = evaluate.load("wer")

processor = SpeechT5Processor.from_pretrained("MBZUAI/artst_asr_v3_qasr")

char2id = defaultdict(lambda: len(char2id))
char2id["[PAD]"]  
char2id["[EOS]"]
char2id["[SOS]"]

with open('_data/diacritics.txt') as f:
    for line in f:
        char2id[line.strip().replace("\n","")]

def encode_text_diacritics(text):
    return [char2id[ch] for ch in text]

def strip_diacritics(text):
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)

# Define the Unicode range for Arabic diacritics

DIACRITIC_PATTERN = re.compile(r'[\u064B-\u0652]')

def extract_diacritics(text):
    diacritic_list = ["[SOS]"]
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if char == ' ':
            diacritic_list.append('_')  # Represent space as '_'
        else:
            diacritics = ''
            j = i + 1
            while j < len(text) and DIACRITIC_PATTERN.match(text[j]):
                diacritics += text[j]
                j += 1
            
            diacritic_list.append(diacritics if diacritics else '-')  # '-' for no diacritic
            i = j - 1  # Move pointer to the last diacritic
        
        i += 1
    diacritic_list.append("[EOS]")
    return diacritic_list

MAX_LENGTH = 256  # example limit for characters

def preprocess_function(example):
    original_text = example["transcription"].strip()
    # 1) Remove diacritics -> input
    input_text = strip_diacritics(original_text)
    # 2) Encode original text diacritics -> labels
    text_diacritics = extract_diacritics(original_text)
    label_ids = encode_text_diacritics(text_diacritics)
    # 3) Get input speech
    if 'audio' in example.keys():
        input_speech = example['audio']['array']
        sr = example['audio']['sampling_rate']
        if sr != 16000:
            input_speech = librosa.resample(input_speech, orig_sr=sr, target_sr=16000)
            sr=16000
    else:
        input_speech, sr = sf.read(example['wav'])
        if sr != 16000:
            input_speech = librosa.resample(input_speech, orig_sr=sr, target_sr=16000)
            sr=16000
        # raise ValueError("No audio found in the example")

    # Truncate or pad if needed
    input_text = input_text[:MAX_LENGTH]
    label_ids = label_ids[:MAX_LENGTH]

    # Convert input_text to IDs for the input side (if desired)
    # For demonstration, we’ll treat them as char2id as well
    input_ids = processor(text= input_text, return_tensors="pt")['input_ids'].squeeze(0)
    # assert input_ids.shape[0] == len(label_ids), f"Input IDs length {input_ids.shape[0]} does not match target length {len(label_ids)}, \n Text length: {len(input_text)}, Diacritics length: {len(text_diacritics)} \n Original text: {original_text} \n Input text: {input_text} \n Diacritics: {text_diacritics}"         
    input_values = processor(audio= input_speech, sampling_rate=sr, return_tensors="pt")['input_values'].squeeze(0)
    example["input_text"] = input_text
    example['diacritics'] = text_diacritics
    example["input_ids"] = input_ids
    example["input_values"] = input_values
    example["labels"] = label_ids
    return example

def filter_unequal_length(example):
    return len(example['input_ids']) == len(example['labels'])


def collate_fn(batch):
    input_text_list = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    input_speech_list = [torch.tensor(x["input_values"], dtype=torch.float) for x in batch]
    label_list = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    max_len = max([len(i) for i in input_text_list])

    padded_inputs = []
    padded_labels = []
    text_attention_masks = []
    for inp, lbl in zip(input_text_list, label_list):
        pad_len = max_len - len(inp)
        inp_padded = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
        lbl_padded = torch.cat([lbl, torch.zeros(pad_len, dtype=torch.long)])
        attn_mask = torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        padded_inputs.append(inp_padded)
        padded_labels.append(lbl_padded)
        text_attention_masks.append(attn_mask)

    speech_max_len = max([len(i) for i in input_speech_list])

    padded_speech = []
    speech_attention_masks = []
    for i in input_speech_list:
        speech_pad_len = speech_max_len - len(i)
        i_padded = torch.cat([i, torch.zeros(speech_pad_len)])
        padded_speech.append(i_padded)
        speech_attention_masks.append(torch.cat([torch.ones(len(i), dtype=torch.long), torch.zeros(speech_pad_len , dtype=torch.long)]))

    text_batch = torch.stack(padded_inputs, dim=0)
    label_batch = torch.stack(padded_labels, dim=0)
    text_attention_mask_batch = torch.stack(text_attention_masks, dim=0)

    speech_batch = torch.stack(padded_speech, dim=0)
    speech_attention_mask_batch = torch.stack(speech_attention_masks, dim=0)

    return text_batch,speech_batch, text_attention_mask_batch, speech_attention_mask_batch, label_batch

# class DiacriticPredictor(nn.Module):
#     def __init__(self, text_model_name='/l/users/hawau.toyin/convert_to_hf/v3_tts_wd', speech_model_name="MBZUAI/artst_asr_v3_qasr", hidden_size=256, num_classes=10):
#         super(DiacriticPredictor, self).__init__()
#         self.text_enc = SpeechT5ForTextToSpeech.from_pretrained(text_model_name).get_encoder()
#         self.speech_enc = SpeechT5ForSpeechToText.from_pretrained(speech_model_name).get_encoder()
#         self.speech_text_cross_attn = nn.MultiheadAttention(embed_dim = self.text_enc.config.hidden_size, num_heads=8, batch_first=True)
#         self.lstm = nn.LSTM(input_size=self.text_enc.config.hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, input_ids, input_values, text_attention_mask, speech_attention_mask):
#         with torch.no_grad():  # Freeze BERT during training
#             if input_ids is not None:
#                 text_emb = self.text_enc(input_ids, attention_mask=text_attention_mask).last_hidden_state
#             speech_emb = self.speech_enc(input_values, attention_mask=speech_attention_mask).last_hidden_state
        
#         if input_ids is not None and text_emb is not None:
#             speech_text_emb, _ = self.speech_text_cross_attn(text_emb, speech_emb, speech_emb)
        

#         lstm_output, _ = self.lstm(speech_text_emb)  # Shape: (batch_size, seq_len, hidden_size * 2)

#         lstm_output = self.dropout(lstm_output)
        
#         logits = self.fc(lstm_output)  # Shape: (batch_size, seq_len, num_classes)
#         return logits

class DiacriticPredictor(nn.Module):
    def __init__(self, text_model_name='/l/users/hawau.toyin/convert_to_hf/v3_tts_wd', speech_model_name="MBZUAI/artst_asr_v3_qasr", hidden_size=256, num_classes=10):
        super(DiacriticPredictor, self).__init__()
        self.text_enc = SpeechT5ForTextToSpeech.from_pretrained(text_model_name).get_encoder()
        self.lstm = nn.LSTM(input_size=self.text_enc.config.hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, input_values, text_attention_mask, speech_attention_mask):
        with torch.no_grad():  # Freeze BERT during training
            if input_ids is not None:
                text_emb = self.text_enc(input_ids, attention_mask=text_attention_mask).last_hidden_state

        lstm_output, _ = self.lstm(text_emb)  # Shape: (batch_size, seq_len, hidden_size * 2)

        lstm_output = self.dropout(lstm_output)
        
        logits = self.fc(lstm_output)  # Shape: (batch_size, seq_len, num_classes)
        return logits
    
def predict_diacritics(model, sentence, audio, device='cuda'):
    model.to(device)
    model.eval()
    input_text = strip_diacritics(sentence)
    text_diacritics = extract_diacritics(sentence)
    input_ids = processor(text= input_text, return_tensors="pt")['input_ids']
    input_values = processor(audio= audio, sampling_rate=16000, return_tensors="pt")['input_values']

    input_ids = input_ids.to(device)
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_ids,input_values, None, None)  # Shape: (1, seq_len, num_classes)

    predictions = torch.argmax(logits, dim=-1).cpu().numpy().flatten()  # Convert to labels

    predictced_diacritics = [id2char[pred] for pred in predictions]
    if '[SOS]' in predictced_diacritics:
        predictced_diacritics.remove('[SOS]')
    if '[EOS]' in predictced_diacritics:
        predictced_diacritics.remove('[EOS]') 
    if '[PAD]' in predictced_diacritics:
        predictced_diacritics.remove('[PAD]')
    text_diacritics.remove('[SOS]') 
    text_diacritics.remove('[EOS]') 
    predicted_sentence = ""
    for i,char in enumerate(input_text): 
        if i >= len(predictced_diacritics): break
        predicted_sentence += char; predicted_sentence += predictced_diacritics[i]
    predicted_sentence = predicted_sentence.replace("_", "").replace("-", "")
    return sentence, predicted_sentence, text_diacritics, predictced_diacritics

criterion = nn.CrossEntropyLoss()


def train_model(model, dataloader, optimizer, num_epochs=25, device='cuda'):
    best_loss = float('inf')
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            text,speech, text_attention_mask,speech_attention_mask, labels = batch
            text,speech, text_attention_mask,speech_attention_mask, labels = text.to(device),speech.to(device), text_attention_mask.to(device),speech_attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(text,speech,text_attention_mask,speech_attention_mask)  # Shape: (batch_size, seq_len, num_classes)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))  # Flatten for loss computation
            loss.backward()
            # if (i + 1) % 4 == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()


            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
        if total_loss < best_loss:
            print(f"Saving model with loss {total_loss}... at epoch {epoch+1}")
            best_loss = total_loss
            torch.save(model.state_dict(), PATH)



if dataset == "TunSwitch":
    train_df = pd.read_csv("/l/users/hawau.toyin/NLP806/TunSwitch/train_cs_manual.csv", sep="\t", skiprows=1, names=['wav','drop','transcription'])
    train_df['wav'] = train_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/TunSwitchCS16K/{x}")
    train_ds = Dataset.from_pandas(train_df)

    test_df = pd.read_csv("/l/users/hawau.toyin/NLP806/TunSwitch/test_cs_manual.csv", sep="\t", skiprows=1, names=['wav','drop','transcription'])
    test_df['wav'] = test_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/TunSwitchCS16K/{x}")
    test_ds = Dataset.from_pandas(test_df)
    train_ds = train_ds.map(preprocess_function, batched=False)

elif dataset == "ClArTTS":
    train_ds = load_dataset("/l/speech_lab/data_806/CLARTTS", split='train')
    test_ds = load_dataset("/l/speech_lab/data_806/CLARTTS", split='test')
    train_ds = train_ds.map(preprocess_function, batched=False)

elif dataset == "MDPC":
    train_ds = load_dataset("/l/speech_lab/data_806/MDPC", split='train')
    test_ds = load_dataset("/l/speech_lab/data_806/MDPC", split='test')
    train_ds = train_ds.map(preprocess_function, batched=False)

elif dataset == "ClArTTS+TunSwitch":
    train_df = pd.read_csv("/l/users/hawau.toyin/NLP806/TunSwitch/train_cs_manual.csv", sep="\t", skiprows=1, names=['wav','drop','transcription'])
    train_df['wav'] = train_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/TunSwitchCS16K/{x}")
    train_ds = Dataset.from_pandas(train_df)
    train_ds_tunswitch = train_ds.map(preprocess_function, batched=False)

    train_ds = load_dataset("/l/speech_lab/data_806/CLARTTS", split='train')
    train_ds_clartts = train_ds.map(preprocess_function, batched=False)

    from datasets import concatenate_datasets
    train_ds = concatenate_datasets([train_ds_tunswitch, train_ds_clartts])

elif dataset == "ClArTTS+TunSwitch+ArVoice":
    train_df = pd.read_csv("/l/users/hawau.toyin/NLP806/TunSwitch/train_cs_manual.csv", sep="\t", skiprows=1, names=['wav','drop','transcription'])
    train_df['wav'] = train_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/TunSwitchCS16K/{x}")
    train_ds = Dataset.from_pandas(train_df)
    train_ds_tunswitch = train_ds.map(preprocess_function, batched=False)

    train_ds = load_dataset("/l/speech_lab/data_806/CLARTTS", split='train')
    train_ds_clartts = train_ds.map(preprocess_function, batched=False)

    train_ds = load_dataset("/l/users/hawau.toyin/ArVoice/ArVoice-h", split='train')
    train_ds_arvoice = train_ds.map(preprocess_function, batched=False)

    from datasets import concatenate_datasets
    train_ds = concatenate_datasets([train_ds_tunswitch, train_ds_clartts, train_ds_arvoice])



train_ds = train_ds.filter(filter_unequal_length)


test_ds = test_ds.map(preprocess_function, batched=False)
# test_ds = test_ds.filter(filter_unequal_length)

print("Train dataset size:", len(train_ds))
# print("Test dataset size:", len(test_ds))


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = DiacriticPredictor(num_classes=len(char2id))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

id2char = {v: k for k, v in char2id.items()}

# if os.path.exists(PATH):
#     model.load_state_dict(torch.load(PATH, weights_only=True))

print("Training model...")

train_model(model, train_loader, optimizer, num_epochs=25, device='cuda')


print("Testing model...")
model.load_state_dict(torch.load(PATH, weights_only=True))
# test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("ASC:")
test_df = pd.read_csv("/l/users/hawau.toyin/NLP806/decoder_model/arzen_test.csv", sep="\t", skiprows=1, names=['wav','gt','drop','transcription'])
test_df['wav'] = test_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ArzEn_SpeechCorpus_1.0/recordings_segmented/{x}")
test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    if 'audio' in example.keys():
        audio = example['audio']['array']
    else: 
        audio, sr = sf.read(example['wav'])
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr=16000
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))
df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_ASC", sep="\t", index=False)


print("ArZen:")
test_df = pd.read_csv("/l/users/hawau.toyin/NLP806/decoder_model/arzen_test.csv", sep="\t", skiprows=1, names=['wav','gt','drop','transcription'])
test_df['wav'] = test_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/ArzEn_SpeechCorpus_1.0/recordings_segmented/{x}")
test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    if 'audio' in example.keys():
        audio = example['audio']['array']
    else: 
        audio, sr = sf.read(example['wav'])
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr=16000
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))
df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_ArzEn", sep="\t", index=False)



print("TunSwitch:")
test_df = pd.read_csv("/l/users/hawau.toyin/NLP806/TunSwitch/test_cs_manual.csv", sep="\t", skiprows=1, names=['wav','drop','transcription'])
test_df['wav'] = test_df['wav'].apply(lambda x: f"/l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/TunSwitchCS16K/{x}")
test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    if 'audio' in example.keys():
        audio = example['audio']['array']
    else: 
        audio, sr = sf.read(example['wav'])
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr=16000
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))
df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_TunSwitch", sep="\t", index=False)

print(f"DER: {100 * cer.compute(references=text_diacritics, predictions=predictced_diacritics)}")
text_diacritics_wo_no_diacritic = [text_diacritic.replace("-", " ").replace("_", " ") for text_diacritic in text_diacritics]
predictced_diacritics_wo_no_diacritic = [predictced_diacritic.replace("-", " ").replace("_", " ") for predictced_diacritic in predictced_diacritics]
print(f"DER wo_no_diacritic symbol: {100 * cer.compute(references=text_diacritics_wo_no_diacritic, predictions=predictced_diacritics_wo_no_diacritic)}")
print(f"Senetence CER w_diacritic: {100 * cer.compute(references=sentences, predictions=predicted_sentences)}")




print("CLARTTS:")
test_ds = load_dataset("/l/speech_lab/data_806/CLARTTS", split='test')
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    audio = example['audio']['array']
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))

df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_CLARTTS", sep="\t", index=False)
print(f"DER: {100 * cer.compute(references=text_diacritics, predictions=predictced_diacritics)}")
text_diacritics_wo_no_diacritic = [text_diacritic.replace("-", " ").replace("_", " ") for text_diacritic in text_diacritics]
predictced_diacritics_wo_no_diacritic = [predictced_diacritic.replace("-", " ").replace("_", " ") for predictced_diacritic in predictced_diacritics]
print(f"DER wo_no_diacritic symbol: {100 * cer.compute(references=text_diacritics_wo_no_diacritic, predictions=predictced_diacritics_wo_no_diacritic)}")
print(f"Senetence CER w_diacritic: {100 * cer.compute(references=sentences, predictions=predicted_sentences)}")


print("ArVoice:")
test_ds = load_dataset("/l/users/hawau.toyin/ArVoice/ArVoice-h", split='test')
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    audio = example['audio']['array']
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))

df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_ArVoice", sep="\t", index=False)

print(f"DER: {100 * cer.compute(references=text_diacritics, predictions=predictced_diacritics)}")
text_diacritics_wo_no_diacritic = [text_diacritic.replace("-", " ").replace("_", " ") for text_diacritic in text_diacritics]
predictced_diacritics_wo_no_diacritic = [predictced_diacritic.replace("-", " ").replace("_", " ") for predictced_diacritic in predictced_diacritics]


print("MDPC:")
test_ds = load_dataset("/l/speech_lab/data_806/MDPC", split='test')
test_ds = test_ds.map(preprocess_function, batched=False)
test_ds = test_ds.filter(filter_unequal_length)
sentences, predicted_sentences, text_diacritics, predictced_diacritics = [], [], [], []
for example in tqdm(test_ds):
    text = example['transcription']
    audio = example['audio']['array']
    sentence, predicted_sentence, text_diacritic, predictced_diacritic = predict_diacritics(model, text, audio, device='cuda')
    sentences.append(sentence)
    predicted_sentences.append(predicted_sentence)
    text_diacritics.append("".join(text_diacritic))
    predictced_diacritics.append("".join(predictced_diacritic))

df = pd.DataFrame({'reference': sentences, 'predicted': predicted_sentences})
df.to_csv(f"_outputs_textOnly/t5_model_{dataset}_MDPC", sep="\t", index=False)

print(f"DER: {100 * cer.compute(references=text_diacritics, predictions=predictced_diacritics)}")
text_diacritics_wo_no_diacritic = [text_diacritic.replace("-", " ").replace("_", " ") for text_diacritic in text_diacritics]
predictced_diacritics_wo_no_diacritic = [predictced_diacritic.replace("-", " ").replace("_", " ") for predictced_diacritic in predictced_diacritics]
# print(f"DER wo_no_diacritic symbol: {100 * cer.compute(references=text_diacritics_wo_no_diacritic, predictions=predictced_diacritics_wo_no_diacritic)}")
# print(f"Senetence CER w_diacritic: {100 * cer.compute(references=sentences, predictions=predicted_sentences)}")


