# # GPU training
# root_path=inbatch
# python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
#     train_batch_neg.py \
#     --device gpu \
#     --save_dir ./checkpoints/${root_path} \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file recall/train.csv


# cpu training
# root_path=inbatch
# python train_batch_neg.py \
#     --device cpu \
#     --save_dir ./checkpoints/${root_path} \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file recall/train.csv 



# 加载simcse训练的模型，模型放在simcse/model_20000
python -u -m paddle.distributed.launch --gpus "1,2,3" \
    ../train_batch_neg.py \
    --device gpu \
    --save_dir /workspace/2208_query_recall/simcse_inbatch_negative/20220819/checkpoints \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 3 \
    --output_emb_size 256 \
    --save_steps 5000 \
    --max_seq_length 64 \
    --margin 0.2 \
    --corpus_file /workspace/2208_query_recall/literature_search_data/recall/corpus.csv \
    --similar_text_pair_file /workspace/2208_query_recall/literature_search_data/recall/dev.csv \
    --train_set_file /workspace/2208_query_recall/bili_data/20220817syn.dic \
    --init_from_ckpt /workspace/2208_query_recall/bili_simcse_model/20220817/checkpoints/model_300000/model_state.pdparams

# 加载post training的模型，模型放在simcse/post_model_10000
# python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
#     train_batch_neg.py \
#     --device gpu \
#     --save_dir ./checkpoints/post_simcse_inbatch_negative \
#     --batch_size 64 \
#     --learning_rate 5E-5 \
#     --epochs 3 \
#     --output_emb_size 256 \
#     --save_steps 10 \
#     --max_seq_length 64 \
#     --margin 0.2 \
#     --train_set_file recall/train.csv  \
#     --init_from_ckpt simcse/post_model_10000/model_state.pdparams
