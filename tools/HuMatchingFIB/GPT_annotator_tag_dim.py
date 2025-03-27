import os
import json
import requests
import re
from tqdm import tqdm
from joblib import Parallel, delayed
from openai import OpenAI
from dotenv import load_dotenv

# 官方号调用
load_dotenv(os.path.expanduser('~/dot_env/openai.env'))

NUM_PROC = 32

def get_response(client, model_name, content, retry_num=2, temperature=0.5, system_prompt=None):

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
    else:
        messages = [
            {"role": "user", "content": content}
        ]
    
    for _ in range(retry_num):  # Retry
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=16384,
                # temperature = temperature,
                response_format={"type": "json_object"},
            )
            res = completion.choices[0].message.content
            return res

        except Exception:
            return None

def process_tag_dim(term, out_json_path, prompt):
    client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY'),
    )
    obj_term = {
        'options':term['reference']['options'],
        'std_ans':term['std_ans'],
        'model_ans':term['model_ans'],
        'segments':term['segmented_text']['segment']
    }
    message = prompt.replace("{Raw_Seg}", json.dumps(obj_term))
    response = get_response(client=client, model_name='gpt-4o-2024-11-20', content=message)
    
    # 格式化输出
    if response is not None:
        try:
            jso_data = json.loads(response)
        except Exception:
            jso_data = None
        
        term['segment_dim'] = jso_data # 存储原始response和解析后的
        line = json.dumps(term, ensure_ascii=False)
        out_file = open(out_json_path, 'a', encoding='utf-8')
        out_file.write(line + '\n')
        out_file.close()
        
def load_prompt_template(template_path):
    """Load the prompt template from a text file."""
    with open(template_path, 'r', encoding='utf-8') as file:
        return file.read()

# segment步骤
def tag_dim_process(input_jsonl_path, output_jsonl_path):
    
    project_path = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(project_path, 'templates/template_DeepSeek_R1_QwQ_segments_classification.txt')
    prompt_template = load_prompt_template(prompt_template_path)
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        data_dict = [json.loads(line) for line in f.readlines()]
        
        if os.path.exists(output_jsonl_path):
            with open(output_jsonl_path, 'r') as f: # 去除已经处理好的，断点续改
                already = set()
                for line in f.readlines():
                    item = json.loads(line)
                    already.add(item['reference']['qid'])
            # 去除已经生成的样本
            New_data_dict = [item for item in data_dict if item['reference']['qid'] not in already]
            data_dict = New_data_dict
        # 多进程并行generate
        Parallel(n_jobs=NUM_PROC)(delayed(process_tag_dim)(entry, output_jsonl_path, prompt_template) for entry in tqdm(data_dict))
            

