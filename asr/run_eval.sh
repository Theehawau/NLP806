# echo "Running evaluation for ArTST v3"

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/artst-v3-clartts/best \
#     --dataset asc,mdpc


echo "Running evaluation for Whisper large"

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/best --dataset asc,mdpc

python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/best --csv_file /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/predictions.csv --use_csv 


# echo "Running evaluation for Whisper tiny"
# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-tiny-clartts/best --dataset asc,mdpc