### 使用方式：

#### 单文件对比：

需要待检测文件命名为{topic}.md
```
python compare.py file \
  --survey path/to/generated.md \
  --human path/to/human.md \
  --output path/to/result.json
```

#### 文件夹对比：
需要待检测文件命名为{topic}.md
```
python compare.py dir \
  --survey_dir path/to/generated_folder \
  --human_dir path/to/human_folder \
  --output path/to/result_folder
```

#### example:
```angular2html
python main.py file --survey "../data/AuotSurvey/3D Object Detection in Autonomous Driving.md" --human "../data/Human/3D Object Detection in Autonomous Driving.md" --output "3_30.json" --logfile "./log_file.txt"
```
