### 使用方式：

#### 文件夹对比：
需要待检测文件命名为{topic}.md，待对比文件夹中文件名应当一一对应相同。
```
python run_quiz_eval.py \
  --survey_dir path/to/generated_folder \
  --human_dir path/to/human_folder \
  --output path/to/result_folder \
  --llm "LLM name" \
  --llm_api_key "llm api key" \
  --llm_api_url "llm url" \
  --emb_model "embedding model" \
  --emb_dimension "embedding model dimension" \
  --emb_api_key "embedding model api key " \
  --emb_api_url "embedding model url" 
```

#### results说明
测试结果会在results/{your_output_dir}下。
- {topic}_specific_results.json中是topic_specific quiz测试得分总览。
- {topic}_compare_results.json为general quiz的测试结果，为保持简洁，其中1代指为human survey，2为输入的待测试survey。