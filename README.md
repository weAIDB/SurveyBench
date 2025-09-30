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
    src/data/{MethodName}/
    ```

2. Human-written reference surveys should be placed under:
    ```
    src/data/HumanSurvey/
    ```

**Requirements:**

- **Filename alignment:**  
For each topic, the `.md` filename must be **identical** between the LLM method directory and `HumanSurvey`.  

    For example:
    ```
    src/data/AutoSurvey/Multimodal Large Language Models.md
    src/data/HumanSurvey/Multimodal Large Language Models.md
    ```


- **Survey format requirement:**  
All survey files must follow a consistent Markdown heading structure.

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
cd src
python run_content_eval.py --mode overall --method AutoSurvey --model gpt-4o-mini --api_key sk_xxx --api_url xxx --output_dir './result/content'
```

**Arguments:**

- `mode` : Evaluation mode. Options:
  - `content` → Evaluate content quality (with/without human reference).
  - `outline` → Evaluate outline quality.
  - `richness` → Evaluate non-textual richness.
  - `overall` → Run all three evaluations (`content with_ref`, `outline`, `richness`).
- `setting` : (Only valid if `--mode content`) Content evaluation setting. Options:
  - `with_ref` → Compare LLM survey against the human-written survey (requires filename alignment).
  - `without_ref_chapter` → Evaluate content chapter by chapter without human reference.
  - `without_ref_document` → Evaluate the whole document without human reference.
    Default: with_ref.
- `--method` : The method name (directory under ./data/) to be evaluated, e.g. AutoSurvey, SurveyForge, LLMxMR-V2.
- `--model` : The model used for evaluation (e.g. gpt-4o-mini).
- `--api_key` : API key for the evaluation model service.
- `--api_url` : API endpoint URL for the evaluation model service.
- `--output_dir` : Directory to save evaluation results (default: ./result/content).



### Quiz-based Evalaution