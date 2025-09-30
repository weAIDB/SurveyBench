<div align="center">
<h1>SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys?</h1>
</div>

## Introduction

<p align="center">
  <img src="./fig/eval_framework.png" width="99%">
</p>

SurveyBench is a fine-grained, quiz-driven evaluation framework, featuring 
(1) typical survey topics source from recent 11,343 arXiv papers and corresponding 4,947 high-quality surveys; 
(2) a multifaceted metric hierarchy that assesses the outline quality (e.g., coverage breadth, logical coherence), content quality (e.g., synthesis granularity, clarity of insights), and non-textual richness;
(3) a dual-mode evaluation protocol that includes content-based and quiz-based answerability tests, explicitly aligned with readers’ informational needs.

## Usage

### Data Prepration

1. Place all generated survey files (`.md`) into the directory:
```
./data/{MethodName}/
```
For example:
```
./data/AutoSurvey/Multimodal Large Language Models.md
./data/SurveyForge/Graph Neural Networks.md
```

2. Human-written reference surveys should be placed under:
```
./data/HumanSurvey/
```
**Requirements:**

- **Filename alignment:**  
For each topic, the `.md` filename must be **identical** between the LLM method directory and `HumanSurvey`.  
For example:
```
./data/AutoSurvey/Multimodal Large Language Models.md
./data/HumanSurvey/Multimodal Large Language Models.md
```


- **Survey format requirement:**  
All survey files must follow a consistent Markdown heading structure, so that sections and subsections can be correctly parsed: 
For example: 
```
# Title
## 1 Introduction
## 2 Section
### 2.1 Subsection
#### 2.1.1 Subsubsection
...
```

### Content-based Evaluation

Here is an example command to evaluate the `AutoSurvey` method on content quality, outline quality and richness.

```bash
python run_eval.py --mode overall --method AutoSurvey --model gpt-4o-mini --api_key sk_xxx --api_url xxx
```
Results will be aggregated and saved in `src/result/content/`.



### Quiz-based Evalaution