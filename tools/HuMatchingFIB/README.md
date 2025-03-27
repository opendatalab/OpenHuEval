## OpenHuEval Reasoning Analysis Tools for HuMatchingFIB

### Overview

This toolkit provides analysis utilities for evaluating the reasoning content of models (QwQ or R1) in the HuMatchingFIB task. It performs segmentation, dimension tagging, and statistical analysis of model reasoning outputs.

### Configuration

Set the following parameters in `main_process.py` before execution:

1. ​**Model**: Target model for analysis  
  eg: Model = 'R1' # Options: 'QwQ' or 'R1'
2. ​**dir_path**: Path to original evaluation results  
  eg: dir_path = r'eval_OpenHuEval_HuMatchingFIB/20250207_185700'
3. ****temp_save_path****: Intermediate result storage patheg: temp_save_path = r'temp_save'

### Execution

Run the analysis pipeline with this command:

```
python main_process.py
```

### Process Flow

The script performs three key operations:

1. ​**Segmentation**: Splits reasoning content into analyzable units
2. ​**Dimension Tagging**: Categorizes reasoning components by predefined dimensions
3. ​**Statistical Analysis**: Generates quantitative metrics for reasoning patterns

Your can check all the result in `temp_save_path`.