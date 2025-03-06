import torch
import librosa
import argparse
import evaluate
import numpy as np

import warnings
warnings.filterwarnings("ignore") #prevent printing of warning messages

from pathlib import Path
from functools import partial
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Dict, List, Union
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer, SpeechT5Processor, SpeechT5Tokenizer, Seq2SeqTrainingArguments


parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, default="clartts,mdpc")
parser.add_argument("--data_path", type=str, default="/l/speech_lab/data_806")
parser.add_argument("--save_dir", type=str, default="./models/artst-v3-qasr-clartts-mdpc")
parser.add_argument("--model_id", type=str, default="mbzuai/artst_asr_v3_qasr")
args = parser.parse_args()

dataset = args.dataset
save_dir = args.save_dir
model_id = args.model_id
data_path = args.data_path

# dataset = "clartts,mdpc"
# save_dir = "./models/artst-v3-qasr-clartts-mdpc"
# model_id = "mbzuai/artst_asr_v3_qasr"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ")


tokenizer = SpeechT5Tokenizer.from_pretrained(model_id)
processor = SpeechT5Processor.from_pretrained(model_id)

# Load dataset
df = load_dataset(data_path)

def prepare_dataset(example):
    # # resample audio with librosa
    if dataset == "clartts":
        if example['sampling_rate'] != 16000:
            audio = librosa.resample(np.array(example['audio']), orig_sr=example['sampling_rate'], target_sr=16000)
        else:
            audio = np.array(example['audio'])
        text = example['text']
    else: 
        if example["audio"]['sampling_rate'] != 16000:
            audio = librosa.resample(np.array(example["audio"]['array']), orig_sr=example["audio"]['sampling_rate'], target_sr=16000)
        else:
            audio = np.array(example["audio"]['array'])
        text = example["transcription"]
    
    example = processor(
        audio=audio,
        sampling_rate=16000,
        text_target=text,
    )
    # # compute input length of audio sample in seconds
    example["input_length"] = len(audio) / 16000

    return example

print("Data Preparation...")
df = df.map(prepare_dataset, remove_columns=df.column_names['train'], num_proc=4)




@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]], padding=True
    ) -> Dict[str, torch.Tensor]:
        batch = {}
        # split inputs and labels since they have to be of different lengths and need different padding methods
        labels_batch = processor.tokenizer.pad({'input_ids':[ sample['labels'] for sample in features]}, return_tensors="pt")
        
        batch['input_values'] = pad_sequence([torch.tensor(sample['input_values'][0]) for sample in features], batch_first=True)
        batch['attention_mask'] = pad_sequence([torch.tensor(sample['attention_mask'][0]) for sample in features], batch_first=True)

        labels = [{"labels": feature["labels"]} for feature in features]


        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )


        batch["labels"] = labels
        return batch
    
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


wer = evaluate.load("wer")
cer = evaluate.load("cer")



def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute metrics
    _wer = 100 * wer.compute(predictions=pred_str, references=label_str)
    _cer = 100 * cer.compute(predictions=pred_str, references=label_str)

    return {"wer": _wer, "cer": _cer}


from transformers import SpeechT5ForSpeechToText

model = SpeechT5ForSpeechToText.from_pretrained(model_id)
model.to(device)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,  # name on the HF Hub
    # auto_find_batch_size=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=6e-5,
    lr_scheduler_type="inverse_sqrt",
    warmup_steps=50,
    max_steps=4000,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=250,
    save_steps=500,
    eval_steps=250,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to="wandb",
    run_name=Path(save_dir).stem,
)


tr_df = df['train'].train_test_split(test_size=0.1)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tr_df["train"],
    eval_dataset=tr_df["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
print("Training...")
trainer.train(resume_from_checkpoint=True)

try:
    trainer.save_model(f"{save_dir}/best/")
except:
    print("Best checkpoint not saved")
    
print("Running Evaluation...")
evaluation = trainer.evaluate(df['test'])
print(evaluation)

print("Running Prediction..")
prediction = trainer.predict(df['test'])

out_file = open(f"{save_dir}/predictions.csv", "w")

print("predictions\treferences", file=out_file)

preds = processor.batch_decode(prediction[0], skip_special_tokens=True)
refs = processor.batch_decode(prediction[1], skip_special_tokens=True)

for i, item in enumerate(preds):
    print(f"{preds[i]}\t{refs[i]}", file=out_file)

out_file.close()

