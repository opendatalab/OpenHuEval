# OpenHuEval: Evaluating Large Language Model on Hungarian Specifics

[![arXiv](https://img.shields.io/badge/arXiv-2503.21500-b31b1b.svg)](https://arxiv.org/abs/2503.21500)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](./LICENSE)

<div align="center">

📃[Paper](https://arxiv.org/abs/2503.21500)
🏰[Project Page](https://opendatalab.github.io/OpenHuEval/)

</div>

## Comparison of related benchmarks

<!-- <style>
.benchmark-table {
    width: 100%;
    border-collapse: collapse;
    text-align: center;
    font-family: Arial, sans-serif;
    margin-top: 20px;
}

.benchmark-table thead {
    background-color: #f1f1f1;
    font-weight: bold;
    position: sticky;
    top: 0; /* 固定表头 */
    z-index: 1; /* 确保表头覆盖内容 */
}

.benchmark-table th, .benchmark-table td {
    border: 1px solid #ddd;
    padding: 8px;
}

/* 表头列样式 */
.benchmark-col {
    text-align: left;
}

.criteria-col {
    width: 12%;
}

/* 表格行样式 */
.benchmark-name {
    text-align: left;
    font-weight: bold;
}

/* 状态：通过 vs. 失败 */
.criteria-result.pass {
    color: green;
    font-weight: bold;
}

.criteria-result.fail {
    color: red;
    font-weight: bold;
}
</style> -->

<section>
    <!-- <h4>Comparison of Related Benchmarks</h4> -->
    <h4></h4>
    <table class="benchmark-table">
        <thead class="fixed-header">
            <tr>
                <th class="benchmark-col">Benchmark</th>
                <th class="criteria-col">Real User Query</th>
                <th class="criteria-col">Self-awareness Evaluation</th>
                <th class="criteria-col">Proverb Reasoning</th>
                <th class="criteria-col">Generative Task & LLM-as-Judge</th>
                <th class="criteria-col">Hungarian Lang</th>
                <th class="criteria-col">Comprehensive Hu-specific</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2406.04770">WildBench</a></td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2411.04368">SimpleQA</a>, <a href="https://arxiv.org/abs/2411.07140">ChineseSimpleQA</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2309.08591">MAPS</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2307.16039">MARC, MMMLU et al.</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2502.07346">BenchMAX</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://aclanthology.org/2023.law-1.19/">MILQA</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://aclanthology.org/2024.lrec-main.733/">HuLU</a></td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result fail">&#x2718;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result fail">&#x2718;</td>
            </tr>
            <tr>
                <td class="benchmark-name"><a href="https://arxiv.org/abs/2503.21500"><strong>OpenHuEval (ours)</strong></td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
                <td class="criteria-result pass">&#x2714;</td>
            </tr>
        </tbody>
    </table>
</section>

## 🛠️ Inference and Evaluation on Opencompass

Below are the steps for quickly downloading OpenHuEval and using OpenCompass for evaluation.

### 1. OpenCompass Environment Setup

Refer to the installation steps for [OpenCompass](https://github.com/open-compass/OpenCompass/?tab=readme-ov-file#%EF%B8%8F-installation).
Please use this [forked repo](https://github.com/jxd0712/opencompass) of OpenCompass for the evaluation of OpenHuEval.

### 2. Download OpenHuEval

```bash
git clone https://github.com/opendatalab/OpenHuEval.git ${path_to_OpenHuEval_repo}

cd ${path_to_opencompass}
mkdir data
ln -snf ${path_to_OpenHuEval_repo} ./data/OpenHuEval
```

### 3. Run Inference and Evaluation

```bash
# use HuSimpleQA task as an example.
cd ${path_to_opencompass}

# modify config file `examples/eval_OpenHuEval_HuSimpleQA.py`: uncomment or add models you want to evaluate
python run.py examples/eval_OpenHuEval_HuSimpleQA.py -r --dump-eval-details
```

The inference and evaluation results would be in `${path_to_opencompass}/outputs`, like this:

```bash
outputs
└── eval_OpenHuEval_HuSimpleQA
    └── 20250312_150000
        ├── predictions # prediction
        │   ├── llama-3_1-8b-instruct-lmdeploy
        │   ├── ...
        │   └── qwen2.5-72b-instruct-lmdeploy
        ├── results # evaluation
        │   ├── llama-3_1-8b-instruct-lmdeploy_judged-by--GPT-4o
        │   ├── ...
        │   └── qwen2.5-72b-instruct-lmdeploy_judged-by--GPT-4o
        └── summary # evaluation summary
            ├── judged-by--GPT-4o-capability_en.csv
            └── judged-by--GPT-4o-capability_hu.csv
```

### 4. Generate Analysis Results

```bash
cd ${path_to_OpenHuEval_repo}

# generate Figure 5 and Figure 6 related statistic result in https://arxiv.org/abs/2503.21500
# Config the related parameters in the tools/HuSimpleQA/lrm_reasoning_process_analysis.py according to the tools/HuSimpleQA/README.md before running.
python tools/HuSimpleQA/lrm_reasoning_process_analysis.py

# generate Figure 9 related statistic result in https://arxiv.org/abs/2503.21500
# Config the related parameters in the tools/HuMatchingFIB/main_process.py according to the tools/HuMatchingFIB/README.md before running.
python tools/HuMatchingFIB/main_process.py
```

## 🖊️ Citation

```bibtex
@inproceedings{yang-etal-2025-openhueval,
    title = "{O}pen{H}u{E}val: Evaluating Large Language Model on {H}ungarian Specifics",
    author = "Yang, Haote  and Wei, Xingjian  and Wu, Jiang  and Ligeti-Nagy, No{\'e}mi  and Sun, Jiaxing  and Wang, Yinfan  and Yang, Gy{\H{o}}z{\H{o}} Zijian  and Gao, Junyuan  and Wang, Jingchao  and Jiang, Bowen  and Wang, Shasha  and Yu, Nanjun  and Zhang, Zihao  and Hong, Shixin  and Liu, Hongwei  and Li, Wei  and Zhang, Songyang  and Lin, Dahua  and Wu, Lijun  and Pr{\'o}sz{\'e}ky, G{\'a}bor  and He, Conghui",
    editor = "Che, Wanxiang  and Nabende, Joyce  and Shutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.390/",
    doi = "10.18653/v1/2025.findings-acl.390",
    pages = "7464--7520",
    ISBN = "979-8-89176-256-5",
}
```

## 💳 License

This project is released under the Apache 2.0 [license](./LICENSE).