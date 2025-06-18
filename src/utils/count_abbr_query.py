"""
统计含有缩写的问题在整个数据集的占比

好的，你提出的初始定义（全大写，长度>=2）是一个不错的起点，但正如我们分析的，它有明显的局限性。为了更准确地统计潜在缩写，我们可以制定一个更细致、更具包容性的定义。
一个更合适的缩写（潜在）定义（用于统计目的）：
我们可以将一个单词（token）视为一个潜在的缩写，如果它满足以下所有基本条件：
    长度: 单词长度至少为 2 个字符。
    字符类型: 主要由字母（A-Z, a-z）和数字（0-9）组成。可以允许内部包含连字符 (-) 或点 (.)，但这会增加复杂性，初步可以先不包含。
    包含大写字母: 单词中必须包含至少一个大写字母。
并且，在此基础上，满足以下至少一个特定模式条件：
    模式 A (多大写): 单词包含两个或更多的大写字母（可以不连续）。
        例子: AIDS, HBsAg, mRNA, ECG, Covid-19 (如果允许连字符), N.gonorrhoeae (如果允许点)。
        理由: 这是最强的缩写信号之一，包括了首字母缩略词和混合大小写的常见缩写。
    模式 B (大写+数字): 单词包含至少一个大写字母和至少一个数字。
        例子: H1N1, COVID19, GABA-A, p53 (如果允许首字母小写)。
        理由: 捕获了包含数字的常见生物医学缩写（如病毒株、基因/蛋白质的变体或家族成员）。
    模式 C (全大写，非罗马): 单词完全由大写字母组成（长度>=2），并且不是一个常见的罗马数字（如 II, III, IV, V, VI, VII, VIII, IX, X...）。
        例子: HIV, DNA, RNA, CBC, EKG。
        理由: 这是你原始定义的核心，但增加了排除罗马数字的条件，以减少假阳性。
这个定义相比你初始定义的优势：
    更广覆盖: 能识别混合大小写 (mRNA)、包含数字 (H1N1) 的缩写，这些在生物医学领域非常常见。
    更少假阳性: 通过排除罗马数字，减少了模式 C 的误报。
    结构化: 通过多个模式组合，逻辑更清晰。
"""
import csv
import json
import random
import re
import string


def simple_word_tokenize(text: str) -> list[str]:
    """
    Very basic word tokenizer based on spaces and punctuation.
    """
    # Convert to lowercase then split? No, keep case for abbreviation detection.
    # Remove punctuation attached to words first (optional, depends on how clean you want it)
    text = text.translate(
        str.maketrans('', '',
                      string.punctuation.replace('-', '').replace(
                          '.', '')))  # Keep hyphen and period for now
    words = text.split()
    # Optional: further clean tokens if needed
    return words


def is_potential_abbreviation(word: str) -> bool:
    """
    检查一个 word 是否是缩写
    """
    # (Use the function defined in the previous answer, acting on the 'word')
    if len(word) < 2:
        return False
    if not re.fullmatch(r'[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*', word):
        return False
    if not any(c.isupper() for c in word):
        return False

    if sum(1 for c in word if c.isupper()) >= 2:
        return True  # Pattern A
    has_upper = any(c.isupper() for c in word)
    has_digit = any(c.isdigit() for c in word)
    if has_upper and has_digit:
        return True  # Pattern B
    if word.isupper():  # Pattern C
        roman_numerals = {
            "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
            "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"
        }
        if word not in roman_numerals:
            return True
    return False


def main():
    """
    主程序入口
    """
    # # --- Your data loading loop ---
    # dataset_file = "./data/dataset/medmcqa/data/dev.jsonl"
    # with open(dataset_file, 'r', encoding='utf-8') as f:
    #     data = f.readlines()

    # query_id = 1
    # query_with_abbr = []
    # for line in data:
    #     item = json.loads(line)
    #     query = item["question"]
    #     options = {
    #         "A": item["opa"],
    #         "B": item["opb"],
    #         "C": item["opc"],
    #         "D": item["opd"]
    #     }
    #     for option_letter, option_text in options.items():
    #         query += " " + option_text

    #     query_words = simple_word_tokenize(query)
    #     if any(is_potential_abbreviation(word) for word in query_words):
    #         query_with_abbr.append(query_id)

    #     query_id += 1

    # print(f"Total questions: {query_id - 1}")
    # print(f"len query_with_abbr: {len(query_with_abbr)}")
    # print(
    #     f"percentage of questions with abbreviations: {len(query_with_abbr) / (query_id - 1) * 100:.2f}%"
    # )

    dataset_file = "./data/dataset/healthfc/healthFC_annotated.csv"
    query_with_abbr = []
    with open(dataset_file, "r", newline="", encoding="utf-8") as f:
        dataset_data = csv.DictReader(f)

        query_id = 1
        for row in dataset_data:
            query = row["en_claim"]
            query_words = simple_word_tokenize(query)
            if any(is_potential_abbreviation(word) for word in query_words):
                query_with_abbr.append(query_id)

            query_id += 1

    print(f"Total questions: {query_id - 1}")
    print(f"len query_with_abbr: {len(query_with_abbr)}")
    print(
        f"percentage of questions with abbreviations: {len(query_with_abbr) / (query_id - 1) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
