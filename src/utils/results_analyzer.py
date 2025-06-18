"""
评估实验结果
"""
import json
import numbers
import os
import random
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.llm_clients import doubao_client
from src.utils import count_abbr_query

llm = doubao_client.DoubaoChat()


def is_negative_query(query: str) -> bool:
    """
    判断一个问题是否含有否定词
    """
    negation_keywords_phrases = [
        r"is not true",
        r"are not true",
        r"not\. true",  # Handles "not. true"
        r"not ture",  # Handles "not ture"
        r"not true",
        r"is false",
        r"are false",
        r"false is",
        r"false statement",
        r"all are true except",
        r"all of these except",
        r"all of these except:",
        r"all of the following are true about .+ except:",
        r"all of the following .* except",
        r"all except",
        r"(is|are|do|does|will|should|has|have|was|were|can|could|may|might)\s+not",
        r"not a\s",
        r"not an\s",
        r"not the\s",
        r"not another",
        r"not involved",
        r"not supplied",
        r"not seen",
        r"not associated",
        r"not related",
        r"not transmitted",
        r"not done",
        r"not belonging",
        r"not helpful",
        r"not present",
        r"not given",
        r"not of\s",
        r"not able",
        r"not require",
        r"not common",
        r"no significance",
        r"least common",
        r"least susceptible",
        r"least useful",
        r"least likely",
        r"except",
        r"false",
        r"NOT",
        r"not",
        r"least",
        r"without",
        r"contraindicated",
        r"cannot",
        r"can not",
        r"never"
    ]

    compiled_negation_patterns = []
    for pattern_str in negation_keywords_phrases:
        if pattern_str.endswith(r"\s"):
            compiled_negation_patterns.append(
                re.compile(r'\b' + pattern_str, re.IGNORECASE))
        else:
            compiled_negation_patterns.append(
                re.compile(r'\b' + pattern_str + r'\b', re.IGNORECASE))

    for preg in compiled_negation_patterns:
        if preg.search(query):
            return True
    return False


def is_abbr_query(query: str) -> bool:
    """
    判断一个问题是否含有缩写词
    """
    query_words = count_abbr_query.simple_word_tokenize(query)
    if any(
            count_abbr_query.is_potential_abbreviation(word)
            for word in query_words):
        return True
    return False


# def is_math_query(options: dict) -> bool:
#     """
#     判断一个问题是否属于数学问题
#     """
#     for value in options.values():
#         for char in value:
#             if char.isdigit():
#                 return True

#     return False


def is_above_query(options: dict) -> bool:
    """
    选项包含 all of the above 或 none of the above
    """
    for value in options.values():
        if "all of the above" in value.lower(
        ) or "none of the above" in value.lower():
            return True

    return False


def is_numeric(s):
    """检查字符串是否可以转换为数字"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def count_numbers_in_text(text):
    """计算文本中独立数字的个数"""
    # 匹配整数和小数，包括可能带单位前缀的数字，但这里我们只关心数字本身
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    return len(numbers)


def contains_math_keywords(text, keywords):
    """检查文本是否包含计算相关的关键词"""
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False


def options_are_primarily_numeric(options_dict):
    """检查选项是否主要为数值或数值范围"""
    numeric_options_count = 0
    total_options = len(options_dict)
    if total_options == 0:
        return False

    for option_text in options_dict.values():
        # 简单检查是否以数字开头或包含数字和常见范围符号
        if re.match(r'^\d+', option_text) or re.search(
                r'\d+-\d+', option_text) or is_numeric(option_text):
            numeric_options_count += 1
    return (numeric_options_count / total_options) >= 0.75  # 假设至少75%的选项是数值型的


def is_computation_query(query: str, options: dict) -> bool:
    """
    判断一个问题是否为计算型问题
    """
    math_keywords = [
        "calculate", "what is the rate", "concentration of", "volume of",
        "clearance", "half-life", "mean", "standard deviation", "SD",
        "prevalence", "confidence interval", "sample size", "ratio",
        "percentage", "average", "how much", "how many", "flow rate", "dose of",
        "eliminated in", "remain after", "expected number", "lower limit",
        "minimum", "total", "per kg", "/kg", "per minute", "/min", "per hour",
        "/hr", "steady state", "first order kinetics"
    ]
    unit_keywords = [
        "mg", "mcg", "ml", "L", "kg", "g", "min", "hour", "hr", "dL", "mEq/L",
        "mmol/L", "mOsm/L", "IU", "U", "drops/min", "mm Hg", "cm", "%", "ppm",
        "mg/dL", "mcg/kg/min", "mL/min", "g/dL"
    ]

    # 条件1: query中包含多个数值
    condition1 = count_numbers_in_text(query) >= 2

    # 条件2: query中包含计算相关的单位或关键词
    all_keywords = math_keywords + unit_keywords
    condition2 = contains_math_keywords(query, all_keywords)

    # 条件3: (这个比较难用纯文本匹配，但可以尝试一些模式)
    # 例如，查找 "what is the rate", "calculate the concentration" 等模式
    # 为了简化，我们主要依赖条件1和2，以及辅助条件4

    # 条件4 (辅助): 选项主要是数值
    condition4_assisting = options_are_primarily_numeric(options)

    # 组合判断逻辑：
    # 核心是问题描述中出现多个数字并且有计算相关的词汇
    if condition1 and condition2:
        return True
    # 或者，问题描述中有强烈的计算指示（例如特定短语），即使数字不多
    # (这里可以添加更复杂的模式匹配，但暂时省略以保持简单)

    # 如果query中数字和关键词不足，但选项强烈暗示是计算结果，也可以考虑
    if condition2 and condition4_assisting:  # 如果有计算词汇且选项是数字
        return True

    # 如果只有多个数字，但没有计算词汇，可能不是计算题
    # 如果只有计算词汇，但没有足够数字，也可能不是（除非选项是数值型）

    return False


def evaluate_cor_and_err():
    """
    评估模型在 results_file1 中回答对的, 在 results_file2 中回答错的 query
    """
    results1_cor_queries = [[], [], []]
    results1_err_queries = [[], [], []]
    results2_cor_queries = [[], [], []]
    results2_err_queries = [[], [], []]

    for attempt in range(1, 4):
        results_file1 = f"./experiments/medmcqa/results/qwen/qwen_ddg_alpha_03_topk_5_results_{attempt}.json"
        results_file2 = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_03_topk_5_results_{attempt}.json"

        with open(results_file1, "r", encoding="utf-8") as f:
            results1 = json.load(f)
        with open(results_file2, "r", encoding="utf-8") as f:
            results2 = json.load(f)

        for item in results1:
            if item["answer"] == item["response"].get("answer"):
                results1_cor_queries[attempt - 1].append(item["id"])
            else:
                results1_err_queries[attempt - 1].append(item["id"])

        for item in results2:
            if item["answer"] == item["response"].get("answer"):
                results2_cor_queries[attempt - 1].append(item["id"])
            else:
                results2_err_queries[attempt - 1].append(item["id"])

    results1_all_attempt_cor = set(results1_cor_queries[0]) & set(
        results1_cor_queries[1]) & set(results1_cor_queries[2])
    results1_all_attempt_err = set(results1_err_queries[0]) & set(
        results1_err_queries[1]) & set(results1_err_queries[2])
    results2_all_attempt_cor = set(results2_cor_queries[0]) & set(
        results2_cor_queries[1]) & set(results2_cor_queries[2])
    results2_all_attempt_err = set(results2_err_queries[0]) & set(
        results2_err_queries[1]) & set(results2_err_queries[2])

    print("results1_all_attempt_cor: ", len(results1_all_attempt_cor))
    print("results1_all_attempt_err: ", len(results1_all_attempt_err))
    print("results2_all_attempt_cor: ", len(results2_all_attempt_cor))
    print("results2_all_attempt_err: ", len(results2_all_attempt_err))

    results1_cor_results2_err = results1_all_attempt_cor & results2_all_attempt_err
    results1_err_results2_cor = results1_all_attempt_err & results2_all_attempt_cor
    print("results1_cor_results2_err len:", len(results1_cor_results2_err))
    print("results1_cor_results2_err:", results1_cor_results2_err)
    print("results1_err_results2_cor len:", len(results1_err_results2_cor))
    print("results1_err_results2_cor:", results1_err_results2_cor)


def sort_ddg_query():
    """
    归类 ddg_query
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    ddg_query_category_file = "./experiments/medmcqa/jsons/doubao_ddg_query_category.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    if os.path.exists(ddg_query_category_file):
        with open(ddg_query_category_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }

    min_id = 1
    max_id = 1000
    for item in query_data:
        if item["id"] < min_id:
            continue
        if item["id"] > max_id:
            break

        query = item["query"]
        option_text = item["options"][item["option"]]
        ddg_query = item["ddg_query"]

        print(f'processing query {item["id"]} option {item["option"]}')
        llm_response = llm.classify_ddg_query_category(query, option_text,
                                                       ddg_query)
        label = int(llm_response["label"])
        results[label].append(f'{item["id"]} {item["option"]}')

        with open(ddg_query_category_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print("len query_data: ", len(query_data))
    for i in range(1, 10):
        print(f"label {i} count: ", len(results[i]))
        print(f"label {i} percentage: ", len(results[i]) / len(query_data))


def count_negative_query_and_accuracy():
    """
    统计否定问题的 accuracy

    指问题中含有 not, except 等词
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    negative_queries = []
    processed_ids = set()
    for item in query_data:
        if item["id"] in processed_ids:
            continue

        query = item["query"]
        if is_negative_query(query):
            negative_queries.append(item["id"])

        processed_ids.add(item["id"])

    print("len negative_queries: ", len(negative_queries))
    # print("random sample negative_queries: ",
    #       random.sample(negative_queries, 10))

    accuracy = []
    for i in range(1, 4):
        results_file = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_07_topk_3_results_{i}.json"

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        cor_count = 0
        for item in results:
            if item["id"] in negative_queries and item["answer"] == item[
                    "response"].get("answer"):
                cor_count += 1

        accuracy.append(cor_count / len(negative_queries) * 100)

    print(f"accuracy: [{', '.join([f'{x:.2f}' for x in accuracy])}]")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy):.2f}")


def count_abbr_query_and_accuracy():
    """
    统计缩写问题的 accuracy

    缩写问题指问题或选项中含有缩写
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    abbr_queries = []
    processed_ids = set()
    for item in query_data:
        if item["id"] in processed_ids:
            continue

        query = item["query"]
        options = item["options"]
        for option_letter, option_text in options.items():
            query += " " + option_text

        if is_abbr_query(query):
            abbr_queries.append(item["id"])

        processed_ids.add(item["id"])

    print("len abbr_queries: ", len(abbr_queries))
    print("random sample abbr_queries: ", random.sample(abbr_queries, 10))

    accuracy = []
    for i in range(1, 4):
        results_file = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_07_topk_3_results_{i}.json"

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        cor_count = 0
        for item in results:
            if item["id"] in abbr_queries and item["answer"] == item[
                    "response"].get("answer"):
                cor_count += 1

        accuracy.append(cor_count / len(abbr_queries) * 100)

    print(f"accuracy: [{', '.join([f'{x:.2f}' for x in accuracy])}]")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy):.2f}")


def count_math_query_and_accuracy():
    """
    统计数学问题的 accuracy

    数学问题指选项为数值
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    math_queries = []
    processed_ids = set()
    for item in query_data:
        if item["id"] in processed_ids:
            continue

        query = item["query"]
        options = item["options"]

        # if is_math_query(query, options):
        #     math_queries.append(item["id"])
        if is_computation_query(query, options):
            math_queries.append(item["id"])

        processed_ids.add(item["id"])

    print("len math_queries: ", len(math_queries))
    print("random sample math_queries: ", math_queries)

    accuracy = []
    for i in range(1, 4):
        results_file = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_07_topk_3_results_{i}.json"

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        cor_count = 0
        for item in results:
            if item["id"] in math_queries and item["answer"] == item[
                    "response"].get("answer"):
                cor_count += 1

        accuracy.append(cor_count / len(math_queries) * 100)

    print(f"accuracy: [{', '.join([f'{x:.2f}' for x in accuracy])}]")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy):.2f}")


def count_above_query_and_accuracy():
    """
    统计 above 问题的 accuracy
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    above_queries = []
    processed_ids = set()
    for item in query_data:
        if item["id"] in processed_ids:
            continue

        query = item["query"]
        options = item["options"]

        if is_above_query(options):
            above_queries.append(item["id"])

        processed_ids.add(item["id"])

    print("len above_queries: ", len(above_queries))
    print("random sample above_queries: ", above_queries)

    accuracy = []
    for i in range(1, 4):
        results_file = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_07_topk_3_results_{i}.json"

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        cor_count = 0
        for item in results:
            if item["id"] in above_queries and item["answer"] == item[
                    "response"].get("answer"):
                cor_count += 1

        accuracy.append(cor_count / len(above_queries) * 100)

    print(f"accuracy: [{', '.join([f'{x:.2f}' for x in accuracy])}]")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy):.2f}")


def count_empty_context_query_and_accuracy():
    """
    统计哪些问题的 context 为空
    """
    min_id = 1
    max_id = 1000

    empty_context_queries = []
    for query_id in range(min_id, max_id + 1):
        summary_file = f"./experiments/medmcqa/scores/ddg_textbook/alpha_03_topk_7/{query_id}_summary.json"

        if not os.path.exists(summary_file):
            empty_context_queries.append(query_id)
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        empty_context = True
        for item in summary_data:
            if item.get("summary") != "":
                empty_context = False
                break
        if empty_context:
            empty_context_queries.append(query_id)

    print("len empty_context_queries: ", len(empty_context_queries))
    # print("empty_context_queries: ", empty_context_queries)

    accuracy = []
    for i in range(1, 4):
        results_file = f"./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_03_topk_7_results_{i}.json"

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        cor_count = 0
        for item in results:
            if item["id"] in empty_context_queries and item["answer"] == item[
                    "response"].get("answer"):
                cor_count += 1

        accuracy.append(cor_count / len(empty_context_queries) * 100)

    print(f"accuracy: [{', '.join([f'{x:.2f}' for x in accuracy])}]")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy):.2f}")


def count_context_source():
    """
    统计 context 中不同来源的数量
    """
    min_id = 1
    max_id = 1000
    count_ddg_source = 0
    count_textbook_source = 0

    for query_id in range(min_id, max_id + 1):
        node_and_neighbors_file = f"./experiments/medmcqa/scores/ddg_textbook/alpha_07_topk_3/{query_id}_node_and_neighbors.json"

        if not os.path.exists(node_and_neighbors_file):
            continue

        with open(node_and_neighbors_file, "r", encoding="utf-8") as f:
            node_and_neighbors_data = json.load(f)

        for item in node_and_neighbors_data:
            node = item["node"]
            neighbors = item["neighbors"]

            if node["source_type"] == "ddg":
                count_ddg_source += 1
            else:
                count_textbook_source += 1

            for neighbor in neighbors:
                if neighbor["source_type"] == "ddg":
                    count_ddg_source += 1
                else:
                    count_textbook_source += 1

    print("count_ddg_source: ", count_ddg_source)
    print("count_textbook_source: ", count_textbook_source)
    print(
        f"ddg_source percentage: {count_ddg_source/(count_ddg_source+count_textbook_source)*100:.2f}",
    )


def count_summary_source():
    """
    统计 summary 中不同来源的数量
    """
    min_id = 1
    max_id = 1000
    count_ddg_source = 0
    count_textbook_source = 0

    for query_id in range(min_id, max_id + 1):
        summary_file = f"./experiments/medmcqa/scores/ddg_textbook/alpha_07_topk_3/{query_id}_summary.json"

        if not os.path.exists(summary_file):
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        for item in summary_data:
            summary = item.get("summary")

            pattern = r'\((.*?)\)'
            matches = re.findall(pattern, summary)

            for content in matches:
                # 去除内容两边的空白字符，以防文件名旁边有空格
                cleaned_content = content.strip()
                if cleaned_content.endswith('.pdf'):
                    count_ddg_source += 1
                elif cleaned_content.endswith('.txt') or \
                    cleaned_content.endswith('.json'):
                    count_textbook_source += 1

    print("count_ddg_source: ", count_ddg_source)
    print("count_textbook_source: ", count_textbook_source)
    print(
        f"ddg_source percentage: {count_ddg_source/(count_ddg_source+count_textbook_source)*100:.2f}",
    )


def get_score_distribution():
    """
    分析不同来源中 sim_score, ner_score 的分布
    """
    # score_dir = "./experiments/medmcqa/scores/ddg_textbook/base_score/"
    # min_id = 1
    # max_id = 1000

    # all_data = []
    # for query_id in range(min_id, max_id + 1):
    #     score_file = f"{score_dir}{query_id}_base_score.json"

    #     if not os.path.exists(score_file):
    #         continue

    #     with open(score_file, "r", encoding="utf-8") as f:
    #         score_data = json.load(f)

    #     all_data.extend(score_data)

    score_dir = "./experiments/medmcqa/scores/ddg_textbook/alpha_07_topk_3/"
    min_id = 1
    max_id = 1000

    all_data = []
    for query_id in range(min_id, max_id + 1):
        score_file = f"{score_dir}{query_id}_node_and_neighbors.json"

        if not os.path.exists(score_file):
            continue

        with open(score_file, "r", encoding="utf-8") as f:
            score_data = json.load(f)

        for item in score_data:
            all_data.append(item["node"])
            for neighbor in item["neighbors"]:
                all_data.append(neighbor)

    # 每个 dict 为一行, 列为 key, 单元格内容为 value
    df = pd.DataFrame(all_data)
    # 设置步长
    bins = np.arange(0, 1.01, 0.1)
    # 获取调色板
    colors = sns.color_palette()

    plt.style.use('seaborn-v0_8-whitegrid')
    for score_type in ["sim_score", "ner_score"]:
        plt.figure(figsize=(12, 7))
        # hue参数用于按source_type分组
        # element="step" 或者 element="poly" 绘制阶梯状或多边形直方图，利于比较
        # multiple="stack" 用于堆叠直方图, multiple="dodge" 用于并列直方图
        # common_norm=False: 每个组独立归一化或计数
        # stat="count": Y轴显示计数; "density": Y轴显示密度; "probability": Y轴显示概率
        sns.histplot(
            data=df,
            x=score_type,
            hue='source_type',
            bins=bins,
            multiple="dodge",
            common_norm=False,
            stat="proportion",
        )

        if score_type == "sim_score":
            score_display = "$Score_{sim}$"
            x_label = f"{score_display}"
        else:
            score_display = "$Score_{ner}$"
            x_label = f"{score_display}"

        plt.xlabel(x_label, fontsize=24)
        plt.ylabel("Frequency", fontsize=24)
        plt.xticks(bins, fontsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.xlim(0, 1)

        unique_sources = sorted(df['source_type'].unique())

        legend_patches = []
        if 'ddg' in unique_sources:
            idx_ddg = unique_sources.index('ddg')
            legend_patches.append(
                mpatches.Patch(color=colors[idx_ddg], label='DuckDuckGo'))
        if 'textbook' in unique_sources:
            idx_textbook = unique_sources.index('textbook')
            legend_patches.append(
                mpatches.Patch(color=colors[idx_textbook], label='Textbook'))

        plt.legend(
            handles=legend_patches,
            loc='upper left',
            fontsize=22,  # 图例项文字大小 (e.g., 'duckduckgo', 'textbook')
            handlelength=4.0,  # 图例颜色格子的长度 (默认约2.0)
            handleheight=1.4,  # 图例颜色格子的高度 (默认约0.7)
        )
        plt.tight_layout()
        plt.show()


def sort_empty_context_query():
    """
    将 context 为空的 query 分类
    """
    min_id = 1
    max_id = 1000

    empty_context_queries = []
    for query_id in range(min_id, max_id + 1):
        summary_file = f"./experiments/medmcqa/scores/ddg_textbook/alpha_03_topk_7/{query_id}_summary.json"

        if not os.path.exists(summary_file):
            empty_context_queries.append(query_id)
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        empty_context = True
        for item in summary_data:
            if item.get("summary") != "":
                empty_context = False
                break
        if empty_context:
            empty_context_queries.append(query_id)

    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    print("len empty_context_queries: ", len(empty_context_queries))

    processed_ids = set()
    results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for item in query_data:
        if item["id"] in processed_ids:
            continue

        if item["id"] in empty_context_queries:
            print(f"query_id: {item['id']}")
            response = llm.classify_empty_context_query_category(
                item["query"], item["options"])
            label = response["label"]

            results[int(label)].append(item["id"])

        processed_ids.add(item["id"])

    for k, v in results.items():
        print(f"label {k} count: ", len(v))
        print(f"label {k} percentage: ", len(v) / len(empty_context_queries))

    with open("./experiments/medmcqa/jsons/empty_context_query_category.json",
              "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    """
    主函数
    """
    # evaluate_cor_and_err()

    # sort_ddg_query()
    # sort_empty_context_query()

    # count_negative_query_and_accuracy()
    # count_abbr_query_and_accuracy()
    # count_math_query_and_accuracy()
    # count_above_query_and_accuracy()
    # count_empty_context_query_and_accuracy()

    # print("context:")
    # count_context_source()
    # print("summary:")
    # count_summary_source()

    get_score_distribution()


if __name__ == "__main__":
    main()
