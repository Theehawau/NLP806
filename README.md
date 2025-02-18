# 1 Diacritization Annotation Interface

## Overview
This repository provides an annotation interface for diacritization using two different models: **Farasa** (online API) and **CATT** (local model). It allows users to process Arabic text, apply diacritization, and manually correct outputs where necessary. 

## Features
- **Diacritization Support**: Uses both the **Farasa** online API or **CATT** local model.
- **Speech Corpus Support**: Processes Arabic-English code-switched speech data.
- **Manual Correction**: Users can manually refine the diacritized output.
- **Data Persistence**: Saves the annotated text to a CSV file.
- **GUI with Gradio**: Provides an interactive interface for easy usage.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- Required libraries: `torch`, `gradio`, `requests`, `pandas`, `pyarabic`, `argparse`

Install dependencies and model via:
```bash
sh setup.sh
```

## Usage
### Running the Interface
To launch the annotation tool, run the following command:
```bash
python ./_code/annotation_interface.py -m /path/to/manifest.csv -r /path/to/audio/files -o /path/to/save/output.csv -s 0 
```

### Arguments
| Argument       | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| `-m, --manifest_file` | Path to the metadata CSV file for the speech corpus.                     |
| `-r, --audio_root` | Path to the directory containing audio files.                            |
| `-o, --out_file` | Path to save the annotated output.                                       |
| `-s, --start_index` | Index to start from, useful for resuming annotation.                     |

## Diacritization Methods
- **Farasa (Online)**: Uses the Farasa API for diacritization.
- **CATT (Local Model)**: Uses a locally trained **TashkeelModel** for diacritization.

### Example API Request (Farasa)
The Farasa API is called via:
```python
url = 'https://farasa.qcri.org/webapi/seq2seq_diacritize/'
api_key = "YOUR_API_KEY"
payload = {'text': text, 'api_key': api_key, 'dialect': "mor"}
data = requests.post(url, data=payload)
if data.ok:
    result = json.loads(data.text)
    print(result["text"])
else:
    print("Error")
```

## GUI Interface (Gradio)
The annotation interface is built using **Gradio** with the following features:
- **Displays audio files** for listening.
- **Automatically diacritizes text** using the selected model.
- **Allows manual correction** of diacritized text.
- **Saves updated text** to the output file.

## Notes
- The CATT model requires a **checkpoint file (`best_ed_mlm_ns_epoch_178.pt`)**. just use the setup.sh file or Download it and place it inside the `catt/models/` directory.
- Ensure that your **API key** for Farasa is valid.
- The system defaults to `CUDA` if available; otherwise, it runs on `CPU`.

-----------------------
# 2 Diacritization Automated Annotation

## **Overview**
This project provides an **automatic annotation** pipeline for **diacritizing Arabic text** while preserving Latin words. The tool processes a dataset containing audio transcriptions, removes Latin words, applies **diacritization using either Farasa or CATT models**, and then reinserts the Latin words in their original positions.

## **Features**
- **Extracts Latin words** from Arabic text and stores them for reinsertion.
- **Diacritizes Arabic text** using:
  - [Farasa Sequence-to-Sequence Diacritizer](https://farasa.qcri.org/webapi/seq2seq_diacritize/)
  - [CATT Model](https://github.com/CATT-Arabic)
- **Restores** Latin words back into their original positions in the sentence.
- **Saves processed data** with new diacritized transcriptions.

## **Installation & Setup**
### **1. Clone the repository & install dependencies**
```bash
sh setup.sh
```

## **Usage**
### **Command-Line Arguments:**
```bash
python ./_code/auto_annotation.py -i <input_file> -o <output_file> -m <catt|farasa> -v <0|1>
```

### **Example Usage:**
```bash
python script.py -i dataset.csv -o output.csv -m farasa -v 0
```

### **Arguments Explained:**
| Argument       | Description |
|---------------|-------------|
| `-i, --manifest_file` | Path to input dataset (CSV file) |
| `-o, --out_file` | Output file to save the processed dataset (default: `./diacritized.csv`) |
| `-m, --module` | Choose diacritization model: `catt` or `farasa` (default: `catt`) |
| `-v, --verbose` | Set verbosity level (`0` for silent, `1` for debug output) |

## **Pipeline Steps**
1. **Read input dataset** (CSV with `ID`, `filename`, `transcription`, `duration` columns).
2. **Extract Latin words** and store them with their positions.
3. **Remove Latin words** and send Arabic text for diacritization.
4. **Apply diacritization** using **Farasa** or **CATT**.
5. **Restore Latin words** in their correct positions.
6. **Save processed dataset** with:
   - `latin_words`: Extracted Latin words with positions.
   - `cleaned_transcription`: Arabic text without Latin words.
   - `annotated_transcription`: Final diacritized Arabic text with Latin words restored.

## **Example Input & Output**
### **Input:**
```
الحركة النسوية كيفاش بدات و علاش بدات و شنوا المطالب متع النسويات ضيفتي ليوم رانية عطافي feminist a book reviewer an english teacher و لfounder متع طبرقة book club و a point كتبت
```
### **Processed Output:**
```
الْحَرَكَةُ النِّسْوِيَّةُ كَيْفَاشٍ بَدَاتٍ وَ عَلَاشٍ بَدَاتٍ وَ شَنُّوا الْمَطَالِبَ مُتَعَ النِّسْوِيَّاتِ ضَيْفَتِي لِيَوْمِ رَانِيَةٍ عَطَافِيٌّ feminist a book reviewer an english teacher وَ لfounder مُتَعَ طَبَرْقَةَ book club وَ a point كَتَبْتُ
```

## **License**
This project is **open-source** under the MIT License.

---

For any questions, feel free to reach out! 🚀
