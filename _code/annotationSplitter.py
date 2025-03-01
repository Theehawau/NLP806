import argparse
import re

import pandas as pd
from tqdm import tqdm


def extract_diacritics(sentence):
    chars = sentence
    diacritic_pattern = re.compile(r'[\u064B-\u0652]')

    diacritics = ""
    for i in range(len(chars)-1):
        if chars[i] ==" ":
            diacritics+="_"
        else:
            if ('\u064B' <= chars[i] <= '\u0652'):
                diacritics += chars[i]
            else:
                if(not diacritic_pattern.match(chars[i + 1])):
                    diacritics += "-"

    ## capturing the last charchater
    # print(f'\\u{ord(chars[-1]):04X}')
    # print (chars[-1] <= '\u064B')
    if chars[-1] == " ":
        diacritics += "_"
    else:
        if ('\u064B' <= chars[-1] <= '\u0652'):
            diacritics += chars[-1]
        else:
            diacritics += "-"
    return diacritics


def remove_diacritics(sentence):
    return re.sub(r'[\u064B-\u0652]', '', sentence)

def process(infilename,outputfilename,separate,remove):

    if not separate and not remove:
        print("No either separate or remove specified")
        return

    df= pd.read_csv(infilename)
    annotations=[]
    transcripts=[]
    if separate:
        for sentence in tqdm(df['annotated_transcription']):
            diacritics = extract_diacritics(sentence)
            annotations.append(diacritics)

        df["annotations_only"] = annotations

    if remove:
        for sentence in tqdm(df['annotated_transcription']):
            transcript = remove_diacritics(sentence)
            transcripts.append(transcript)

        df["transcripts_only"] = transcripts

    df.to_csv(outputfilename, index=False)



# # Tests
# sentence = "مُحمَّدٌ "
# sentence2= "فَمَا حَاجَة"
# sentence3= "فَمَا حَاجَة للْولَاد أَيْ كِيْفَاه"
# diacritics_only = extract_diacritics(sentence2)
# print(diacritics_only)  # Output: "ُ َّ ٌ"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--manifest_file", type=str, help="Path to manifest file")
    parser.add_argument("-o", "--out_file", type=str, help="Path to save the output", default="./output.csv")
    parser.add_argument("-s", "--separate", type=int, help="separate annotation in new column choose 1",default=0)
    parser.add_argument("-r", "--remove", type=int, help="remove annotations choose 1",default=0)

    args = parser.parse_args()
    manifest_file = args.manifest_file
    out_file = args.out_file
    separate = args.separate
    remove = args.remove
    process(manifest_file, out_file,separate,remove )