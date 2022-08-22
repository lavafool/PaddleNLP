CUDA_VISIBLE_DEVICES=2 python ../feature_extract.py \
        --model_dir /workspace/2208_query_recall/bili_simcse_model/20220817/output \
        --batch_size 128 \
        --device "gpu" \
        --corpus_file "/workspace/2208_query_recall/bili_data/20220817query_pool.dic" \
        --out_dir "/workspace/2208_query_recall/bili_data/20220817bili_simcse_embedding"

