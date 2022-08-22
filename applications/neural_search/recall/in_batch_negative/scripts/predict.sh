# gpu version

root_dir="/workspace/2208_query_recall/simcse_inbatch_negative/20220819/checkpoints" 
python -u -m paddle.distributed.launch --gpus "3" \
    ../predict.py \
    --device gpu \
    --params_path "${root_dir}/model_10000/model_state.pdparams" \
    --output_emb_size 256 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "/workspace/2208_query_recall/bili_data/20220817syn.dic" \
    --output_file "/workspace/2208_query_recall/bili_data/20220817syn_score.dic"


# cpu
# root_dir="checkpoints/inbatch" 
# python predict.py \
#     --device cpu \
#     --params_path "${root_dir}/model_40/model_state.pdparams" \
#     --output_emb_size 256 \
#     --batch_size 128 \
#     --max_seq_length 64 \
#     --text_pair_file "recall/test.csv"
