### 使用方式：

#### 文件夹对比：
需要待检测文件命名为{topic}.md，待对比文件夹中文件名应当一一对应相同。
```
python run_quiz_eval.py \
  --survey_dir path/to/generated_folder \
  --human_dir path/to/human_folder \
  --output path/to/result_folder
```

#### example:
```angular2html
python run_quiz_eval.py --survey_dir "../data/for_test/Example_Tool/" --human_dir "../data/for_test/HumanSurvey/" --output_dir "results_as_example"
```
