# python ro_vs_eng/filter_jql.py \
#     --input_dir /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql \
#     --output_dir /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92 \
#     --min_score_gemma 3.568359375 \
#     --min_score_mistral 2.5546875 \
#     --min_score_llama 2.99609375

# python small_model_inference/json_joiner.py \
#     --input_dir /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92 \
#     --output_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92.json

# python small_model_inference/parquet.py \
#     --input_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92.json \
#     --output_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92.parquet

# python small_model_inference/token_stats.py \
#     --in_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92.json \
#     --out_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_jql_filtered92_stats.json

# python small_model_inference/ground.py

# python small_model_inference/token_analysis.py --in_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_filtered_joined/token_stats_2.json

# python full_dataset_analysis/filtered_counts.py --input_file /export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_filtered_joined/filtered_3_5.json --output_file filtered_full.json

python dataset_upload.py
