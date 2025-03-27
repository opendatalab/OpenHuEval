import os
import json

# 存储抽取的QwQ的raw reasoning content
def get_raw_pred_QwQ(dir_path, temp_save_path):

    raw_root = dir_path + r'/results/QwQ-32B-Preview/OpenHuEval_HuMatchingFIB_250205-prompt_en.json'
    with open(raw_root, 'r', encoding='utf-8') as f:
        Data = json.load(f)['details']

    if not os.path.exists(temp_save_path): # 判断暂存文件夹是否存在
        os.makedirs(temp_save_path)
    
    save_raw_path = temp_save_path + r'/QwQ_raw_predition_all.jsonl'
    if not os.path.exists(save_raw_path):
        with open(save_raw_path, 'w', encoding='utf-8') as f:
            for k, v in Data.items():
                v['prediction'] = v.pop('raw_pred')
                f.write(json.dumps(v, ensure_ascii=False) + "\n")

# 存储抽取的R1的raw reasoning content
def get_raw_pred_R1(dir_path, temp_save_path):

    raw_root = dir_path + r'/predictions/deepseek_r1_api_sensetime'
    jso_list = os.listdir(raw_root)
    
    map_raw = {} # 记录qid和reasoning content
    jso_list = [item for item in jso_list if 'prompt_en' in item]
    for jso in jso_list:
            f_name = os.path.join(raw_root, jso)
            with open(f_name, 'r', encoding='utf-8') as f_read:
                raw_data = json.load(f_read)
                for k, v in raw_data.items():
                    map_raw[v['gold']['qid']] = v['prediction']

    with open(dir_path + r'/results/deepseek_r1_api_sensetime/OpenHuEval_HuMatchingFIB_250205-prompt_en.json', 'r', encoding='utf-8') as f:
        result = json.load(f)['details']

    result = [item for k, item in result.items()]
    if not os.path.exists(temp_save_path): # 判断暂存文件夹是否存在
        os.makedirs(temp_save_path)
    
    save_raw_path = temp_save_path + r'/R1_raw_predition_all.jsonl'
    if not os.path.exists(save_raw_path):
        with open(save_raw_path, 'w', encoding='utf-8') as f:
            for term in result: # 将每个样本的reasoning content 附加到对应的result中
                term['prediction'] = map_raw[term['reference']['qid']]
                f.write(json.dumps(term, ensure_ascii=False) + "\n")

# 选择具体的model
def get_raw_pred(Model, dir_path, temp_save_path):
    if Model == "QwQ":
        get_raw_pred_QwQ(dir_path, temp_save_path)
    elif Model == "R1":
        get_raw_pred_R1(dir_path, temp_save_path)