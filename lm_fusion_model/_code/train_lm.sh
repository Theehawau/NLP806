TEXT=/l/users/hawau.toyin/NLP806/lm_fusion_model/data
# fairseq-preprocess \
#     --trainpref $TEXT/tashkeela_train.txt \
#     --testpref $TEXT/tashkeela_test.txt \
#     --validpref $TEXT/tashkeela_valid.txt \
#     --destdir $TEXT/tashkeela \
#     --workers 12

mkdir -p /l/users/hawau.toyin/NLP806/lm_fusion_model/models/tashkeela_LM

fairseq-train --task language_modeling \
  /l/users/hawau.toyin/NLP806/lm_fusion_model/data/tashkeela \
  --save-dir /l/users/hawau.toyin/NLP806/lm_fusion_model/models/tashkeela_LM \
  --arch transformer_lm \
  --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 \
  --sample-break-mode none \
  --max-tokens 1024 \
  --update-freq 32 \
  --fp16 \
  --max-update 300000 