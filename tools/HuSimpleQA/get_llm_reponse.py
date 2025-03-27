import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
from tqdm import tqdm

import os.path as osp
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    openai = None

load_dotenv(osp.expanduser("~/dot_env/proxy_on.env"))
load_dotenv(osp.expanduser("~/dot_env/openai.env"))


def get_response(client,model_name,query,retry_num=3,temperature=0.5):

    for _ in range(retry_num):  # Retry
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                        {"role": "system", "content":"You are a helpful assistant."},
                        {"role": "user", "content": query},
                    ],

                temperature = temperature,
                response_format={"type": "json_object"},
            )
            res = completion.choices[0].message.content
            if res[-1]!="}":
                return None
            res = json.loads(completion.choices[0].message.content)
            return res
        except Exception as e:
            print(f"Error processing question: {e}")
            time.sleep(10)
    return None


def get_processed_result(qs, client, model,queries, processed_indices,worker_nums=5):
    failed_idx = []
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=worker_nums) as executor:
        future_to_question = {
            executor.submit(get_response, client, model,queries[idx]): idx
            for idx, q in enumerate(qs)}
        for future in tqdm(as_completed(future_to_question), total=len(future_to_question)):
            idx = future_to_question[future]
            try:
                result = future.result()
                if result:
                    results[idx] = result
                else:
                    failed_idx.append({"idx": idx, "error": "Processing failed"})
            except Exception as e:
                failed_idx.append({"idx": idx, "error": str(e)})

    if failed_idx:
        pprint(failed_idx)
    return results









