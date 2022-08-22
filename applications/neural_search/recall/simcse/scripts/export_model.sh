CUDA_VISIBLE_DEVICES=2 python ../export_model.py \
    --params_path /workspace/2208_query_recall/bili_simcse_model/20220817/checkpoints/model_300000/model_state.pdparams \
    --output_path=/workspace/2208_query_recall/bili_simcse_model/20220817/output \
    --output_emb_size 256