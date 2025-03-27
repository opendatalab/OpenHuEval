import json
import os
import re
import time
import traceback
from collections import OrderedDict, defaultdict

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from get_llm_reponse import get_processed_result
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


def save_jsonl(data, output_file, mode="w"):
    with open(output_file, mode, encoding="utf-8") as f:
        for idx, data in enumerate(data):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_processed_indices(path, id_col):
    processed_indices = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                processed_indices.add(q.get(id_col))
    return processed_indices


prompts = {}
prompt_path = "tools/HuSimpleQA/templates"
for prompt in os.listdir(prompt_path):
    with open(os.path.join(prompt_path, prompt), "r") as f:
        prompts[prompt.replace(".txt", "")] = f.read()
lang_map = {"en": "English", "zh": "Chinese", "hu": "Hungarian"}


def get_data(input_file):
    processed_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data.values():
        jso = {"qid": d["gold"]["qid"], "prediction": d["prediction"], "gold": d["gold"]}
        processed_data.append(jso)
    return processed_data


def get_labled_data(input_file):
    labeled_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    for d in data:
        exps = set(d["shift_expression"])

        # 提供转折词标记
        phrase_counter = {phrase: 0 for phrase in exps}
        if len(exps) == 0:
            jso = d.copy()
            text = d["prediction"]
            jso["labeld_prediction"] = text
            jso["shift_words"] = {}
            jso["shift_words"]["word"] = ""
            labeled_data.append(jso)

        for exp in exps:
            text = d["prediction"]

            def replace_func(match):
                # 对于每个匹配的词组，增加对应的计数
                phrase_counter[exp] += 1
                count = phrase_counter[exp]
                # 返回格式化后的词组
                return f"<shift_word_{count}>{match.group(0)}<shift_word_{count}>"

            text = re.sub(rf"\b{re.escape(exp)}\b", replace_func, text)
            jso = d.copy()
            jso["labeld_prediction"] = text
            jso["shift_words"] = {}
            jso["shift_words"]["word"] = exp
            labeled_data.append(jso)
    return labeled_data


def get_extract_shift_expression_res(input_data, client, model, processed_indices, worker_nums):
    evaluate_query = []
    for idx, data in enumerate(input_data):
        question = data["gold"]["question"]
        prediction = data["prediction"]
        prompt = prompts["shift_expression"]
        query = prompt.replace("<question>", question).replace("<answer>", prediction)
        evaluate_query.append(query)
    evaluate_res = get_processed_result(input_data, client, model, evaluate_query, processed_indices, worker_nums=worker_nums)
    return evaluate_res


def get_confirm_shift_expression_res(input_data, client, model, processed_indices, worker_nums):
    evaluate_query = []
    for idx, data in enumerate(input_data):
        # question = data["gold"]["question"]
        text = data["labeld_prediction"]
        prompt = prompts["confirm_shift_postion"]
        query = prompt.replace("<input_text>", text)
        evaluate_query.append(query)
    evaluate_res = get_processed_result(input_data, client, model, evaluate_query, processed_indices, worker_nums=worker_nums)
    return evaluate_res


def get_access_res(input_data, client, model, processed_indices, worker_nums):
    evaluate_query = []
    temp_data = []
    for idx, data in enumerate(input_data):
        split_solution = data["split_prediction"]
        question = data["question"]
        std_ans = data["answer"]
        for solution in split_solution:
            prompt = prompts["access_thought"]
            query = prompt.replace("<problem>", question)
            query = query.replace("<split_solutions>", solution)
            query = query.replace("<expected_answer>", std_ans)
            evaluate_query.append(query)
            temp_data.append(data)
    evaluate_res = get_processed_result(temp_data, client, model, evaluate_query, processed_indices, worker_nums=worker_nums)
    return evaluate_res


def extract_shift_expression(compass_output_folder, model_name, work_dir, client, model, batch_size, worker_nums):
    input_folder = os.path.join(compass_output_folder, "predictions", model_name)
    output_file = os.path.join(work_dir, "with_shift_expression_hu.jsonl")
    input_data = []
    files = os.listdir(input_folder)
    for file in files:
        input_file = os.path.join(input_folder, file)
        input_data += get_data(input_file)

    processed_indices = read_processed_indices(output_file, "qid")
    total_num = len(input_data)
    # total_num = 100
    retry_num = 0
    while total_num == processed_indices or retry_num <= 3:
        for batch in range(0, total_num, batch_size):
            try:
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{start_time}:  Processing batch {batch} to {batch + batch_size}")
                batch_data = input_data[batch : batch + batch_size]
                processed_indices = read_processed_indices(output_file, "qid")
                batch_data = [x for x in batch_data if x.get("qid") not in processed_indices]
                if batch_data == []:
                    continue
                evaluate_res = get_extract_shift_expression_res(batch_data, client, model, processed_indices, worker_nums)

                for idx, data in enumerate(batch_data):
                    data["pred_lan"] = evaluate_res[idx]["language"]
                    data["shift_expression"] = evaluate_res[idx]["shift_expression"]
                save_jsonl(batch_data, output_file, mode="a")
            except Exception as e:
                print(f"Error processing batch {batch} to {batch + batch_size}: {e}")
                traceback.print_exc()
                time.sleep(10)
                continue
        retry_num += 1
        processed_indices = read_processed_indices(output_file, "qid")


def confirm_shift_expression(work_dir, client, model, batch_size, worker_nums):
    input_file = os.path.join(work_dir, "with_shift_expression_hu.jsonl")
    output_file = os.path.join(work_dir, "shift_postion_hu.jsonl")

    input_data = get_labled_data(input_file)

    processed_indices = read_processed_indices(output_file, "qid")
    total_num = len(input_data)
    # total_num = 100
    retry_num = 0
    while total_num == processed_indices or retry_num <= 3:
        for batch in range(0, total_num, batch_size):
            try:
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{start_time}:  Processing batch {batch} to {batch + batch_size}, total_num: {total_num}")
                batch_data = input_data[batch : batch + batch_size]
                processed_indices = read_processed_indices(output_file, "qid")
                batch_data = [x for x in batch_data if x.get("qid") not in processed_indices]
                if batch_data == []:
                    continue
                evaluate_res = get_confirm_shift_expression_res(batch_data, client, model, processed_indices, worker_nums)

                final_res = []
                for idx, data in enumerate(batch_data):
                    if not evaluate_res[idx]:
                        continue
                    data["shift_words"]["idxs"] = evaluate_res[idx]["shift_word_ids"]
                    final_res.append(data)
                save_jsonl(final_res, output_file, mode="a")
            except Exception as e:
                print(f"Error processing batch {batch} to {batch + batch_size}: {e}")
                traceback.print_exc()
                time.sleep(10)
                continue
        retry_num += 1
        processed_indices = read_processed_indices(output_file, "qid")


def get_split_idx(jso):
    """
    获取每个转折词的位置
    """
    text = jso["labeld_prediction"]
    text = text.replace("\n\n", " ")
    text = text.split(" ")
    word = jso["shift_words"]["word"]
    idxs = jso["shift_words"]["idxs"]

    idx_list = []
    for idx in idxs:
        patterns = [
            f"<shift_word_{idx}>{word}<shift_word_{idx}>",
            f"<shift_word_{idx}>{word}<shift_word_{idx}>,",
        ]
        for pattern in patterns:
            if pattern in text:
                idx_list.append(text.index(pattern))
                break
    return idx_list


def get_split_text(text, split_idx):
    """
    获取分割后的文本
    """
    split_text = []
    split_idx.sort()
    text = text.replace("\n\n", " ")
    text = text.split(" ")
    l, r = 0, 0
    for idx in split_idx:
        r = idx
        split_text.append(" ".join(text[l:r]))
        l = r
    if r < len(text):
        split_text.append(" ".join(text[r:]))
    split_text = [x for x in split_text if x]
    return split_text


def split_answer(work_dir):
    input_file = os.path.join(work_dir, "shift_postion_hu.jsonl")
    output_file = os.path.join(work_dir, "split_answer_hu.jsonl")
    with open(input_file, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    idx_split_dict = {}
    # get split idxs for each shift expression
    for d in data:
        idx = d["qid"]
        split_idxs = get_split_idx(d)
        if idx not in idx_split_dict:
            idx_split_dict[idx] = split_idxs
        else:
            idx_split_dict[idx].extend(split_idxs)

    # split answer
    with open(output_file, "w", encoding="utf-8") as fout:
        processed_id = set()
        for d in data:
            qid = d["qid"]
            if qid in processed_id:
                continue
            split_idxs = idx_split_dict[qid]
            answer = d["prediction"]
            split_prediction = get_split_text(answer, split_idxs)
            jso = d["gold"].copy()
            jso["prediction"] = d["prediction"]
            # jso["shift_phrase"] = list(set(d["shift_expression"]))
            jso["pred_language"] = d["pred_lan"]
            jso["split_prediction"] = split_prediction
            fout.write(json.dumps(jso, ensure_ascii=False) + "\n")
            processed_id.add(qid)


def get_split_data(input_file):
    split_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        split_data = [json.loads(line) for line in f]
    return split_data


def access(work_dir, client, model, batch_size, worker_nums):
    input_file = os.path.join(work_dir, "split_answer_hu.jsonl")
    output_file = os.path.join(work_dir, "split_answer_access_hu.jsonl")
    input_data = get_split_data(input_file)

    processed_indices = read_processed_indices(output_file, "qid")
    total_num = len(input_data)
    # total_num = 100
    retry_num = 0
    while total_num == processed_indices or retry_num <= 3:
        for batch in range(0, total_num, batch_size):
            try:
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{start_time}:  Processing batch {batch} to {batch + batch_size}")
                batch_data = input_data[batch : batch + batch_size]
                processed_indices = read_processed_indices(output_file, "qid")
                batch_data = [x for x in batch_data if x.get("qid") not in processed_indices]
                if batch_data == []:
                    continue
                evaluate_res = get_access_res(batch_data, client, model, processed_indices, worker_nums)

                print(f"Processed {len(evaluate_res)} samples")

                res_idx = 0
                for idx, data in enumerate(batch_data):
                    split_num = len(data["split_prediction"])
                    confident_scores = []
                    for i in range(split_num):
                        confident_scores.append(evaluate_res[res_idx]["confident_score"])
                        res_idx += 1
                    data["confident_scores"] = confident_scores
                save_jsonl(batch_data, output_file, mode="a")
            except Exception as e:
                print(f"Error processing batch {batch} to {batch + batch_size}: {e}")
                traceback.print_exc()
                time.sleep(10)
                continue
        retry_num += 1
        processed_indices = read_processed_indices(output_file, "qid")


def read_predictions(jsons_path):
    # 读取完整的样本
    file_list = os.listdir(jsons_path)
    results = []
    for file in file_list:
        if "_hu_" in file:
            with open(os.path.join(jsons_path, file), "r", encoding="utf-8") as file:
                results += list(json.load(file).values())
    new_data = {}
    for answer in results:
        new_data[answer["gold"]["qid"]] = answer
    return new_data


def read_thought(jsons_path):
    # 读取完整的样本
    with jsonlines.open(jsons_path) as reader:
        results = list(reader)
    new_data = {}
    for answer in results:
        new_data[answer["qid"]] = answer
    return new_data


def merge_json(tokenizer_path, work_dir, model_name, compass_output_folder):
    # 读取相关的数据
    predictions_dict_path = os.path.join(compass_output_folder, "predictions", model_name)
    results_dict_path = os.path.join(compass_output_folder, "results", model_name + "_judged-by--GPT-4o-2024-08-06")
    thought_dict_path = os.path.join(work_dir, "split_answer_access_hu.jsonl")
    # 构建保存中间结果的路径
    output_file_path = os.path.join(work_dir, "merged_info.jsonl")

    predictions_dict = read_predictions(predictions_dict_path)
    results_dict = read_predictions(results_dict_path)
    thought_dict = read_thought(thought_dict_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    final_result = []
    for key, values in tqdm(thought_dict.items()):
        thought_number = len(thought_dict[key]["split_prediction"])
        correct_thought_number = thought_dict[key]["confident_scores"].count("2")
        correct_thought_ratio = np.around(correct_thought_number / thought_number, 1)
        final_result.append(
            {
                "qid": key,
                "answer_tokens_length": len(tokenizer.encode(predictions_dict[key]["prediction"])),
                "thought_number": thought_number,
                "correct_thought_ratio": correct_thought_ratio,
                "judge_results": json.loads(results_dict[key]["prediction"].replace("json\n", "")[3:-3].strip())["evaluation"],
            }
        )
    with jsonlines.open(output_file_path, mode="w") as writer:
        writer.write_all(final_result)


# 需要两个指标，一个作为x轴，另一个座位y轴
def counting_answer_v1(final_result, type_key, type_value):
    toknes_count = defaultdict(list)
    # 分类
    if type_key == "confidence_score":
        for item in final_result:
            if item[type_key] == -1:
                new_key = -1
            else:
                new_key = item[type_key] // 10
            toknes_count[new_key].append(item[type_value])
    else:
        for item in final_result:
            toknes_count[item[type_key]].append(item[type_value])

    # 统计
    toknes_count = OrderedDict(sorted(toknes_count.items(), key=lambda item: item[0]))
    avg_tokens_count = {}
    if type_value == "qid":
        for key, value in toknes_count.items():
            avg_tokens_count[key] = int(len(value))
    else:
        for key, value in toknes_count.items():
            avg_tokens_count[key] = np.around(sum(value) / len(value), 2)

    return avg_tokens_count


# 只对一类指标进行统计，统计这个指标的分布情况
def counting_answer_v2(final_result, type_key):
    total_num = 0
    result_correct = defaultdict(int)
    result_total = defaultdict(int)
    # 999 代表没有出现正确的thought
    for item in final_result:
        # 只统计有正确thought的
        if item["judge_results"] == "INCORRECT":
            total_num += 1
            # 统计总的数量
            for ind in range(item["thought_number"]):
                max_len = 10
                if int(ind) + 1 > max_len:
                    result_total[11] += 1
                else:
                    result_total[int(ind) + 1] = result_total[int(ind) + 1] + 1
            # 统计正确的数量
            for ind in item[type_key]:
                max_len = 10
                if int(ind) + 1 > max_len:
                    result_correct[11] += 1
                else:
                    result_correct[int(ind) + 1] = result_correct[int(ind) + 1] + 1
    result_correct = OrderedDict(sorted(result_correct.items(), key=lambda item: int(item[0])))
    result_ratio = defaultdict(int)
    for key, value in result_correct.items():
        result_ratio[key] = np.round(value / result_total[key], 2)
    return result_total, result_correct, result_ratio


# 只对一类指标进行统计，统计这个指标的分布情况 统计INCORRECT类型
def counting_answer_v3(final_result, type_key):
    total_num = 0
    result_count = defaultdict(int)
    # 999 代表没有出现正确的thought
    for item in final_result:
        # 只统计有正确thought的
        if item["judge_results"] == "INCORRECT":
            total_num += 1
            for ind in range(10):
                if item[type_key] > ind / 10:
                    result_count[ind / 10] += 1
                else:
                    break
    result_count = OrderedDict(sorted(result_count.items(), key=lambda item: int(item[0])))
    new_result_count = defaultdict(int)
    for key, value in result_count.items():
        new_result_count[key] = np.round(value / total_num, 2)
    return result_count, new_result_count


# 只对一类指标进行统计，统计这个指标的分布情况 统计NOT_ATTEMPTED类型
def counting_answer_v4(final_result, type_key):
    total_num = 0
    result_count = defaultdict(int)
    # 999 代表没有出现正确的thought
    for item in final_result:
        # 只统计有正确thought的
        if item["judge_results"] == "NOT_ATTEMPTED":
            total_num += 1
            for ind in range(10):
                if item[type_key] > ind / 10:
                    result_count[ind / 10] += 1
                else:
                    break
    result_count = OrderedDict(sorted(result_count.items(), key=lambda item: int(item[0])))
    new_result_count = defaultdict(int)
    for key, value in result_count.items():
        new_result_count[key] = np.round(value / total_num, 2)
    return result_count, new_result_count


def plot_bar(deep_seek_tokens1, qwq_tokens1, deep_seek_tokens2, qwq_tokens2, output):
    # 准备
    real_names = ["CORRECT", "INCORRECT", "NOT ATTEMPTED"]
    x_ticks = ["Deepseek-R1", "QwQ"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.2))  # 2行1列的子图布局
    # 调用bar绘制
    correct1 = [deep_seek_tokens1[0], qwq_tokens1[0]]
    incorrect1 = [deep_seek_tokens1[1], qwq_tokens1[1]]
    not_attempet1 = [deep_seek_tokens1[2], qwq_tokens1[2]]
    x = range(2)
    width = 0.3
    axes[0].bar(x, correct1, width=width, color=(173 / 255.0, 23 / 255.0, 40 / 255.0), label=real_names[0])
    axes[0].bar([i + width for i in x], incorrect1, width=width, color=(38 / 255.0, 87 / 255.0, 173 / 255.0), label=real_names[1])
    axes[0].bar([i + width * 2 for i in x], not_attempet1, width=width, color=(1 / 255.0, 142 / 255.0, 1 / 255.0), label=real_names[2])
    axes[0].set_ylabel("Generated Tokens", fontsize=12)
    axes[0].set_xticks([i + width for i in x], x_ticks, fontsize=12)
    axes[0].set_yticks(ticks=[0, 1000, 2000])
    axes[0].legend()

    # 调用bar绘制
    correct2 = [deep_seek_tokens2[0], qwq_tokens2[0]]
    incorrect2 = [deep_seek_tokens2[1], qwq_tokens2[1]]
    not_attempet2 = [deep_seek_tokens2[2], qwq_tokens2[2]]
    x = range(2)
    width = 0.3
    axes[1].bar(x, correct2, width=width, color=(173 / 255.0, 23 / 255.0, 40 / 255.0), label=real_names[0])
    axes[1].bar([i + width for i in x], incorrect2, width=width, color=(38 / 255.0, 87 / 255.0, 173 / 255.0), label=real_names[1])
    axes[1].bar([i + width * 2 for i in x], not_attempet2, width=width, color=(1 / 255.0, 142 / 255.0, 1 / 255.0), label=real_names[2])
    axes[1].set_ylabel("Number of Thoughts", fontsize=12)
    axes[1].set_xticks([i + width for i in x], x_ticks, fontsize=12)
    axes[1].set_yticks(ticks=[0, 5, 10, 15])
    axes[1].legend()

    # 添加图例
    plt.tight_layout()
    plt.savefig(output, format="pdf", dpi=300)
    print(f"{output} saved")


def plot_line(x_label, datas, output, xl, yl):
    fontsize = 12
    # 创建画布和轴
    plt.figure(figsize=(6, 4))  # 设置图形大小
    markers = ["o", "o", "s", "s", "o", "o"]  # 区分不同类型
    colors = ["#AB252D", "#AB252D", "#AB252D", "#AB252D", "#2956A9", "#2956A9"]  # 区分不同数据集
    linestyles = ["-", "--", "-", "--", "-", "--"]  # 区分不同模型
    for ind, row in enumerate(datas.itertuples()):
        plt.plot(x_label, row[2:], label=row[1], color=colors[ind], marker=markers[ind], linestyle=linestyles[ind])  # 模型1的折线

    # 设置标题和坐标轴标签
    plt.xlabel(xl, fontsize=fontsize)  # x轴标签
    plt.ylabel(yl, fontsize=fontsize)  # y轴标签
    plt.yticks(ticks=[0, 0.4, 0.8])
    # 添加图例
    # plt.legend(fontsize=10)
    plt.savefig(output, format="pdf", dpi=300)
    plt.savefig(output.replace("pdf", "svg"), format="svg", dpi=300)
    print(f"{output} saved")


def plot_thought_number_and_answer_tokens_length(work_folder, deep_seek_output_file_path, qwq_output_file_path):
    with jsonlines.open(deep_seek_output_file_path) as reader:
        deep_seek_final_result = list(reader)
    with jsonlines.open(qwq_output_file_path) as reader:
        qwq_seek_final_result = list(reader)
    # Answer
    models_name = ["deep_seek", "qwq"]
    modle_answers = {"deep_seek": deep_seek_final_result, "qwq": qwq_seek_final_result}

    x_names = ["judge_results"]
    y_names = ["answer_tokens_length", "thought_number"]

    for model_name in models_name:
        for x_name in x_names:
            results = []
            for y_name in y_names:
                results.append(counting_answer_v1(modle_answers[model_name], x_name, y_name))
            df = pd.DataFrame(results, y_names)
            df.to_csv(os.path.join(work_folder, f"{model_name}_{x_name}_{y_name}.csv"), index=True, encoding="utf-8")
    # 绘制第一个图片 每个模型生成的Token数量
    deep_seek_result1 = pd.read_csv(os.path.join(work_folder, "deep_seek_judge_results_thought_number.csv")).iloc[0][1:]
    qwq_result1 = pd.read_csv(os.path.join(work_folder, "qwq_judge_results_thought_number.csv")).iloc[0][1:]
    deep_seek_result2 = pd.read_csv(os.path.join(work_folder, "deep_seek_judge_results_thought_number.csv")).iloc[1][1:]
    qwq_result2 = pd.read_csv(os.path.join(work_folder, "qwq_judge_results_thought_number.csv")).iloc[1][1:]
    output = os.path.join(work_folder, "Analysis_on_HuSimpleQA.pdf")
    plot_bar(deep_seek_result1, qwq_result1, deep_seek_result2, qwq_result2, output)


def plot_correct_thought_ratio(work_folder, deep_seek_output_file_path, qwq_output_file_path):
    with jsonlines.open(deep_seek_output_file_path) as reader:
        deep_seek_final_result = list(reader)
    with jsonlines.open(qwq_output_file_path) as reader:
        qwq_seek_final_result = list(reader)

    models_name = ["deep_seek", "qwq"]
    modle_answers = {"deep_seek": deep_seek_final_result, "qwq": qwq_seek_final_result}

    #  统计每个问题的正确thought占这个问题中所有thought的比例 统计比例大于0，大于0.1 等等的数量
    x_name = "correct_thought_ratio"
    results = []
    label_names = ["IN_Deepseek-R1", "IN_QwQ"]
    for model_name in models_name:
        conut_result, raito_result = counting_answer_v3(modle_answers[model_name], x_name)
        results.append(raito_result)
    df_INCORRECT = pd.DataFrame(results, label_names)
    #  统计每个问题的正确thought占这个问题中所有thought的比例 统计比例大于0，大于0.1 等等的数量
    x_name = "correct_thought_ratio"
    results = []
    label_names = ["NA_Deepseek-R1", "NA_QwQ"]
    for model_name in models_name:
        conut_result, raito_result = counting_answer_v4(modle_answers[model_name], x_name)
        results.append(raito_result)
    df_NOT_ATTEMPTED = pd.DataFrame(results, label_names)
    # 从其他论文中获取的数据
    other_results = {
        0.0: {"NA_Deepseek-R1*": 0.72, "NA_QwQ*": 0.78},
        0.1: {"NA_Deepseek-R1*": 0.55, "NA_QwQ*": 0.67},
        0.2: {"NA_Deepseek-R1*": 0.38, "NA_QwQ*": 0.59},
        0.3: {"NA_Deepseek-R1*": 0.22, "NA_QwQ*": 0.48},
        0.4: {"NA_Deepseek-R1*": 0.11, "NA_QwQ*": 0.32},
        0.5: {"NA_Deepseek-R1*": 0.05, "NA_QwQ*": 0.24},
        0.6: {"NA_Deepseek-R1*": 0.05, "NA_QwQ*": 0.16},
        0.7: {"NA_Deepseek-R1*": 0.05, "NA_QwQ*": 0.10},
        0.8: {"NA_Deepseek-R1*": 0, "NA_QwQ*": 0.10},
        0.9: {"NA_Deepseek-R1*": 0, "NA_QwQ*": 0.02},
    }
    df_INCORRECT_OTHERS = pd.DataFrame(other_results)
    pd.concat([df_INCORRECT, df_NOT_ATTEMPTED, df_INCORRECT_OTHERS], axis=0).to_csv(os.path.join(work_folder, "correct_thought_ratio.csv"), index=True, encoding="utf-8")

    yl = "Distribution"
    xl = "Ratio of Correct Thoughts"
    x_label = [">0", "≥0.1", "≥0.2", "≥0.3", "≥0.4", "≥0.5", "≥0.6", "≥0.7", "≥0.8", "≥0.9"]
    data = pd.read_csv(os.path.join(work_folder, "correct_thought_ratio.csv"))
    output = os.path.join(work_folder, "Analysis_on_HuSimpleQA2.pdf")
    plot_line(x_label, data, output, xl, yl)


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tool_model_name = "gpt-4o-2024-05-13"
    worker_nums = 5
    batch_size = 100
    # 设置模型的Tokenizer路径
    deepseek_tokenizer_path = "your_ocheckpoints/models-deepseek-ai-DeepSeek-R1"
    qwq_tokenizer_path = "your_ocheckpoints/models-Qwen-QwQ-32B-Preview"
    # opcompass 输出的路径，这里需要注意要具体到数据集和某个日期
    compass_output_folder = "your_opencompass_outputs/eval_OpenHuEval_HuSimpleQA/20250212_170313"
    # 用于保存过程中的数据
    work_folder = "tmp"
    # 需要处理的模型的名称列表
    model_names = ["deepseek_r1_api_aliyun", "QwQ-32B-Preview"]

    for model_name in model_names:
        # 针对每个模型构建独立的路径
        work_model_folder = os.path.join(work_folder, model_name)
        os.makedirs(work_model_folder, exist_ok=True)

        # 1. 读取LLM的Response， 并且划分成thoughts
        extract_shift_expression(compass_output_folder, model_name, work_model_folder, client, tool_model_name, batch_size, worker_nums)

        # 2. 确定那些expression是真正的转折词
        confirm_shift_expression(work_model_folder, client, tool_model_name, batch_size, worker_nums)

        # 3. 根据转折词的索引将Response划分成thoughts
        split_answer(work_model_folder)

        # 4. 判断每一个thougnt是否正确
        access(work_model_folder, client, tool_model_name, batch_size, worker_nums)

        # 5. 合并所有相关的文件 然后保存到一个文件中 并统计相关数据
        # 因为这里要统计Response的Token数量，因此这里需要读取对应模型的Tokenizer
        if "deepseek" in model_name:
            tokenizer_path = deepseek_tokenizer_path
        elif "qwq" in model_name:
            tokenizer_path = qwq_tokenizer_path

        deep_seek_final_result = merge_json(tokenizer_path, work_model_folder, model_name, compass_output_folder)

    # 6. 绘制并保存图片
    deep_seek_output_file_path, qwq_output_file_path = [os.path.join(work_folder, model_name, "merged_info.jsonl") for model_name in model_names]
    plot_thought_number_and_answer_tokens_length(work_folder, deep_seek_output_file_path, qwq_output_file_path)
    plot_correct_thought_ratio(work_folder, deep_seek_output_file_path, qwq_output_file_path)
