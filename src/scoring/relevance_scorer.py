"""
(qa dataset) 计算 chunk 的 relevance score
"""
import json
import os
import warnings
from itertools import combinations

import numpy as np
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter

import settings
from src.utils import logger_config

warnings.simplefilter("ignore", category=FutureWarning)

# 尝试使用 GPU，如果 GPU 不可用则回退到 CPU
spacy.prefer_gpu()
SIM_NLP = spacy.load("en_core_sci_lg")
NER_BIONLP13CG = spacy.load("en_ner_bionlp13cg_md")
NER_BC5CDR = spacy.load("en_ner_bc5cdr_md")

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
)

logger = logger_config.get_logger(__name__)


def extract_entities(text: str) -> set:
    """
    使用 spaCy 模型提取实体
    """
    doc1 = NER_BIONLP13CG(text)
    doc2 = NER_BC5CDR(text)

    entities = set()
    for ent in doc1.ents:
        entities.add(ent.text)
    for ent in doc2.ents:
        entities.add(ent.text)

    return entities


def calc_query_chunk_sim_score(query: str, chunk: str) -> float:
    """
    计算 query 和 chunk 的余弦相似度

    Args:
        query: 查询文本
        chunk: 文档文本
    
    Returns:
        float: 余弦相似度
    """
    doc1 = SIM_NLP(query)
    doc2 = SIM_NLP(chunk)
    return doc1.similarity(doc2)


def calc_chunks_sim_score(chunks: list[str]) -> np.ndarray:
    """
    计算 chunk 互相之间的 cosine similarity

    Args:
        chunks: list[str]，包含所有句子的列表

    Returns:
        np.ndarray: 对称的相似度矩阵, shape为(len(chunks), len(chunks))
    """
    n = len(chunks)
    similarity_matrix = np.eye(n)  # 对角线为1（自己与自己的相似度）

    # 预处理所有句子
    docs = [SIM_NLP(chunk) for chunk in chunks]

    # 计算所有两两组合的相似度
    for i, j in combinations(range(n), 2):
        similarity = docs[i].similarity(docs[j])
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity  # 矩阵是对称的

    return similarity_matrix


def calc_query_chunk_ner_score(query: str, chunk: str) -> float:
    """
    计算 query 和 chunk 之间的 ner_score
    """
    # 提取实体集合
    query_entities = extract_entities(query)
    chunk_entities = extract_entities(chunk)

    # 计算匹配分数
    ner_matches = len(query_entities & chunk_entities)
    total_query_entities = len(query_entities)

    return ner_matches / total_query_entities if total_query_entities > 0 else 0.0


def find_content_list(root_dir: str) -> list[str]:
    """
    (Helper) 从 root_dir 中找到 content_list.json

    Args:
        root_dir: 目录
    
    Returns:
        list[str]: content_list.json 的路径列表
    """
    content_list_paths = []

    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith("content_list.json"):
                file_path = os.path.join(dir_path, file_name)
                content_list_paths.append(file_path)

    return content_list_paths


def collect_pdf_text(content_file: str) -> str:
    """
    (Helper) 收集 pdf 文本
    """
    pdf_text = ""

    logger.info("processing content file: %s", content_file)
    with open(content_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    for item in content:
        if item["type"] != "text":
            continue
        pdf_text += item["text"] + " "

    return pdf_text


def process_doc_ddg(item: dict, save_dir: str):
    """
    对 query_id 的问题, 处理来自 ddg 的 pdf 文本
    """
    results = []

    # 组合 query 和 options 用于后续计算 score
    query_id = item["id"]
    query = item["query"]
    options = item["options"]
    query_and_options = query
    for option_letter, option_text in options.items():
        query_and_options += f" {option_letter}: {option_text}"

    # 找出 pdf 文本划分为 chunk
    pdf_dir = f"./experiments/medmcqa/docs/{query_id}/"
    if not os.path.exists(pdf_dir):
        logger.info("%s not exists, skip...", pdf_dir)
        return

    content_files = find_content_list(pdf_dir)
    for content_file in content_files:
        pdf_text = collect_pdf_text(content_file)
        chunks = TEXT_SPLITTER.split_text(pdf_text)

        for chunk in chunks:
            sim_score = calc_query_chunk_sim_score(query_and_options, chunk)
            ner_score = calc_query_chunk_ner_score(query_and_options, chunk)
            # rel_score = ALPHA * sim_score + (1 - ALPHA) * ner_score

            results.append({
                "id": query_id,
                "query": query,
                "options": options,
                "chunk": chunk,
                "source_type": "ddg",
                "source": content_file,
                "sim_score": sim_score,
                "ner_score": ner_score,
                # "rel_score": rel_score,
            })

    # list[dict] 为空
    if not results:
        logger.info("%d results is empty", query_id)
        return

    # 使用流式写入替代列表缓存
    rel_score_file = os.path.join(save_dir, f"{query_id}_base_score.json")
    with open(rel_score_file, "w", encoding="utf-8") as f:
        f.write("[")
        for i, record in enumerate(results):
            if i > 0:
                f.write(",")
            json.dump(record, f, ensure_ascii=False, indent=4)
        f.write("]")


def process_doc_textbook(item: dict, save_dir: str):
    """
    对 query_id 的问题, 处理来自 textbook 的文本
    """
    textbook_file = "./experiments/medmcqa/jsons/doubao_en_textbooks.json"
    with open(textbook_file, "r", encoding="utf-8") as f:
        textbook_data = json.load(f)

    # 组合 query 和 options 用于后续计算 score
    query_id = item["id"]
    query = item["query"]
    options = item["options"]
    query_and_options = query
    for option_letter, option_text in options.items():
        query_and_options += f" {option_letter}: {option_text}"

    # 最后的结果文件
    rel_score_file = os.path.join(save_dir, f"{query_id}_base_score.json")
    if os.path.exists(rel_score_file):
        with open(rel_score_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # 处理 textbook 的文本
    for textbook_item in textbook_data:
        if textbook_item["id"] != query_id:
            continue
        if textbook_item["textbooks"] is None:
            continue

        for textbook in textbook_item["textbooks"]:
            paths = settings.EN_TEXTBOOKS_CATEGORY_AND_PATH[textbook]
            for path in paths:
                logger.info("processing textbook: %s", path)
                with open(path, "r", encoding="utf-8") as f:
                    entities_and_chunks = json.load(f)

                # 找出 query 中的实体，匹配 textbook 中的句子
                query_and_options_entities = extract_entities(query_and_options)
                for entity in query_and_options_entities:
                    # chunks 是一个 list[str]
                    chunks = entities_and_chunks.get(entity, [])

                    for chunk in chunks:
                        similarity = calc_query_chunk_sim_score(
                            query_and_options, chunk)
                        ner_score = calc_query_chunk_ner_score(
                            query_and_options, chunk)

                        results.append({
                            "id": query_id,
                            "query": query,
                            "options": options,
                            "chunk": chunk,
                            "source_type": "textbook",
                            "source": path,
                            "sim_score": similarity,
                            "ner_score": ner_score,
                        })

    # 使用流式写入替代列表缓存
    with open(rel_score_file, "w", encoding="utf-8") as f:
        f.write("[")
        for i, record in enumerate(results):
            if i > 0:
                f.write(",")
            json.dump(record, f, ensure_ascii=False, indent=4)
        f.write("]")


def calc_base_score_ddg():
    """
    计算 chunk 对应的 sim_score 和 ner_score
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    processed_ids = set()
    for item in query_data:
        query_id = item["id"]
        if query_id < min_id or query_id in processed_ids:
            continue
        if query_id > max_id:
            break
        logger.info("processing query %d...", query_id)

        # 处理来自 ddg 的 pdf 文本
        save_dir = "./experiments/medmcqa/scores/ddg/base_score/"
        process_doc_ddg(item, save_dir)

        processed_ids.add(query_id)
        logger.info("query %d processed", query_id)


def calc_base_score_ddg_textbook():
    """
    计算 base score, 包括 ddg 和 textbook 来源
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    processed_ids = set()
    for item in query_data:
        query_id = item["id"]
        if query_id < min_id or query_id in processed_ids:
            continue
        if query_id > max_id:
            break
        logger.info("processing query %d...", query_id)

        # 预先运行 calc_base_score_ddg(), 并复制一份
        # 处理来自 textbook 的 pdf 文本, 追加形式写入
        save_dir = "./experiments/medmcqa/scores/ddg_textbook/base_score/"
        process_doc_textbook(item, save_dir)

        processed_ids.add(query_id)
        logger.info("query %d processed", query_id)


def main():
    """
    主程序入口
    """
    # calc_base_score_ddg()
    calc_base_score_ddg_textbook()


if __name__ == "__main__":
    main()
