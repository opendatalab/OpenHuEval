## OpenHuEval Reasoning Analysis Tools for HuSimpleQA

### Introduction
This project contains one Python script designed to analyze and evaluate the reasoning capabilities of Large Language Models (LLMs) on Hungarian Factual Open-ended Qustions tasks. 

### File Descriptions
1. get_llm_response.py
This script uses the OpenAI API to get responses.
2. lrm_reasoning_process_anslysis.py
This script segments, classifies, and provides statistical analysis of model responses.

### Usage:
```
python lrm_reasoning_process_analysis.py
```

### Configuration:  
- Requires OpenAI API key (loaded from .env file) 
- Customizable API base URL 
- Tool model name
- Adjustable number of worker threads for parallel processing
- Adjustable batch size
- Set model tokenizer path, compass output path and work folder




