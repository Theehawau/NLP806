import re
import os
import random
import unicodedata

def filter_unequal_length(example):
    return len(example['input_ids']) == len(example['labels'])


def strip_diacritics(text):
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)

def replace_arabic_punctuation_with_space(text):
    # List of Arabic punctuation marks 
    arabic_punctuation = "،؛.؟,\/#!$%\^&\*«;:»=‫‏‮‬‎\-'_`»!~()\{\}\[\]\""
    
    # Create a pattern to match any of the Arabic punctuation marks
    pattern = re.compile(f"[{re.escape(arabic_punctuation)}]")
    
    # Replace matched punctuation marks with space
    cleaned_text = pattern.sub("", text)
    
    # Return the cleaned text
    return cleaned_text

# Define the Unicode range for Arabic diacritics
DIACRITIC_PATTERN = re.compile(r'[\u064B-\u0652]')

def extract_diacritics(text):
    diacritic_list = []
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
    
    return diacritic_list

with open('_data/diacritics.txt') as f:
    ACCEPTED_DIACRITICS = [x.replace('\n','') for x in f.readlines()]

def extract_diacritics_(text):
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

def strip_diacritics(text):
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)

def split_data(data_raw):
    data_new = list()
    
    for line in data_raw:
        line = line.replace('.', '.\n')
        line = line.replace(',', ',\n')
        line = line.replace('،', '،\n')
        line = line.replace(':', ':\n')
        line = line.replace(';', ';\n')
        line = line.replace('؛', '؛\n')
        line = line.replace('(', '\n(')
        line = line.replace(')', ')\n')
        line = line.replace('[', '\n[')
        line = line.replace(']', ']\n')
        line = line.replace('{', '\n{')
        line = line.replace('}', '}\n')
        line = line.replace('«', '\n«')
        line = line.replace('»', '»\n')
        
        for sub_line in line.split('\n'):
            if len(strip_diacritics(sub_line).strip()) == 0:
                continue
            
            if len(strip_diacritics(sub_line).strip()) > 0 and len(strip_diacritics(sub_line).strip()) <= 400:
                data_new.append(sub_line.strip())
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    if len(strip_diacritics(tmp_line).strip()) + len(strip_diacritics(word).strip()) + 1 > 400:
                        if len(strip_diacritics(tmp_line).strip()) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        if tmp_line == '':
                            tmp_line = word
                        else:
                            tmp_line += ' '
                            tmp_line += word
                if len(strip_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())

    return data_new



if __name__ == "__main__":
    DATASET_PATH = '../lm_fusion_model/data/all_texts.txt'
    train_raw = ''
    with open(DATASET_PATH, 'r') as file:
        train_raw = file.readlines()
    
    train_split = split_data(train_raw)    
    
    print('Training examples (split):', len(train_split))
    train, test =  random.sample(train_split, int(0.8 * len(train_split))), random.sample(train_split, int(0.2 * len(train_split)))
    out_file = "_data/tashkeela_train.txt"
    with open(out_file, 'w') as f:
        for sample in set(train):
            f.write(replace_arabic_punctuation_with_space(sample) + '\n')
            
    out_file = "_data/tashkeela_test.txt"
    with open(out_file, 'w') as f:
        for sample in set(test):
            f.write(replace_arabic_punctuation_with_space(sample) + '\n')
    
    text = " ".join(train_split)
    # extract unique diacritcs 
    diacritics = extract_diacritics(text)
    print("Diacritics:      ", set(diacritics))
    print("Diacritics count:", len(set(diacritics)))

    out_file = "_data/diacritics_tashkeela.txt"
    with open(out_file, 'w') as f:
        for diacritic in set(diacritics):
            f.write(diacritic + '\n')
            
    print("Data split and diacritics extraction completed!")