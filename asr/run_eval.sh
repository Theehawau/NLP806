# echo "Running evaluation for ArTST v3: ClarTTS,mdpc"

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/artst-v3-clartts-mdpc/best \
#     --dataset clartts,asc,mdpc

# echo "Running evaluation for ArTST v3_qasr: ClarTTS,mdpc"

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/artst-v3-qasr-clartts-mdpc/best \
#     --dataset clartts,asc,mdpc

# echo "Running evaluation for Whisper large"

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/best --dataset asc,mdpc

# python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/best --csv_file /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts/predictions.csv --use_csv 


echo "Running evaluation for Whisper large: ClarTTS,mdpc"
python eval.py --model_name /l/users/hawau.toyin/NLP806/asr/models/whisper-large-clartts-mdpc/best --dataset clartts,asc,mdpc