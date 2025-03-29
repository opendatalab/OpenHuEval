import os
import json
import requests
import re
from tqdm import tqdm
from joblib import Parallel, delayed
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
from get_raw_prediction import *
from GPT_annotator_segment import *
from GPT_annotator_tag_dim import *
from reasoning_stat import *

if __name__ == "__main__":
    
    # 0. 输入待处理的模型和路径
    Model = 'QwQ' # QwQ, R1
    dir_path = r'eval_OpenHuEval_HuMatchingFIB/20250207_185700' # response result path
    temp_save_path = r'temp_save'
    
    # 1. 从原始的respone中抽取raw prediction
    get_raw_pred(Model, dir_path, temp_save_path)
    
    # 2. 根据raw prediction进行segment
    raw_pred_path = temp_save_path + '/' + Model + '_raw_predition_all.jsonl'
    save_seg_path = temp_save_path + '/' + Model + '_raw_predition_all_segment.jsonl'
    seg_process(raw_pred_path, save_seg_path)
    
    # 3. 根据segment进行tag dim
    dim_save_path = temp_save_path + '/' + Model + '_raw_predition_all_segment_dim.jsonl'
    tag_dim_process(save_seg_path, dim_save_path)
    
    # 4. 根据dim统计相应指标
    output_statistic_path = temp_save_path + '/' + Model + '_respone_dim_statistic_result'
    analyze_reasoning_stats(dim_save_path, output_statistic_path)