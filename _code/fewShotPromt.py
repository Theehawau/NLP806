
import openai
import json
import pandas as pd

## just to keep the OPENAI_API_KEY in seperate file not part of git
from license import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

prompt={
  "task": "Diacritize the following Tunisian dialect transcriptions accurately. Preserve foreign words as they are and ensure proper diacritization of Arabic words.",
  "instructions": [
    "Apply proper diacritization to each Arabic word.",
    "Preserve Tunisian pronunciation patterns, including shortened vowels and consonant clusters.",
    "Use sukoon (ْ) when necessary, especially for consonant endings.",
    "Maintain foreign words as they are (e.g., English or French words should not be diacritized).",
    "Output the response in fully diacritized Arabic text."
  ],
  "examples": [
    {
      "input": "و شنوا المطالب متع النسويات ضيفتي",
      "output": "وْ شْنُوَا المَطَالِبْ مْتَاعْ النِّسْوِيَّاتْ ضَيْفْتِي"
    }
  ],
  "data": [
  ],
  "output_format": "JSON"
}

def promptChatGPT(prompt):
    # Send request to OpenAI GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in Tunisian Arabic dialect diacritization."},
            {"role": "user", "content": json.dumps(prompt)}
        ],
        temperature=0.2
    )
    return response

#inputfile="../l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/language_annotation/dev_cs.csv"
inputfile="../l/speech_lab/CodeSwitchedDataset[code_switched_dataset]/TunSwitch/language_annotation/test_cs.csv"

df=pd.read_csv(inputfile)
dicData=df[["ID","transcription"]].to_dict('records')
start=0
step=10
for i in range (start,len(dicData),step):
    prompt["data"]=[dicData[i:i+step]]
    response = promptChatGPT(prompt)
    result = response.choices[0].message.content.strip("```json").strip("```").strip()
    df2=(pd.DataFrame(json.loads(result)))
    mergeddf=df.iloc[i:i+step]
    mergeddf=mergeddf.reset_index()
    mergeddf['ID2']=df2[df2.columns[0]]
    mergeddf['annotated_transcription']=df2[df2.columns[1]]
    mergeddf.to_csv("output/test/test_cs_"+str(i)+".csv",index=False)
    del df2, mergeddf


