python ../../export_to_serving.py \
    --dirname "/workspace/2208_query_recall/bili_simcse_model/20220817/output" \
    --model_filename "inference.get_pooled_embedding.pdmodel" \
    --params_filename "inference.get_pooled_embedding.pdiparams" \
    --server_path "./serving_server" \
    --client_path "./serving_client" \
    --fetch_alias_names "output_embedding"
    
# python web_service.py