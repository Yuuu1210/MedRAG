"""
预先处理 textbook, 构建 inverted_index 或 faiss_index
"""
import json
import os

import faiss
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import settings
from src.document import qa_calc_score
from src.utils import logger_config

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=256,  # 每个 chunk 不超过 256 个字符
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
)

logger = logger_config.get_logger(__name__)


def build_textbook_inverted_index():
    """
    每个 textbook 构建 inverted_index 并保存到文件
    """
    textbook_dir = "./data/dataset/medqa/textbooks/en/"
    output_dir = "./data/textbooks/en_inverted_index/"

    # 找到所有 textbooks
    textbooks = []
    for file in os.listdir(textbook_dir):
        file_path = os.path.join(textbook_dir, file)
        textbooks.append(file_path)
    logger.info("find %d textbooks", len(textbooks))

    # 读取 textbook 并分块
    for textbook in textbooks:
        logger.info("processing textbook: %s", textbook)
        entities_and_chunks = {}

        with open(textbook, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = TEXT_SPLITTER.split_text(text)

        for chunk in chunks:
            # 提取实体
            entities = qa_calc_score.extract_entities(chunk)
            for entity in entities:
                if entity not in entities_and_chunks:
                    entities_and_chunks[entity] = []
                entities_and_chunks[entity].append(chunk)

        # 保存倒排索引
        file_name = os.path.join(
            output_dir,
            os.path.basename(textbook).replace(".txt", ".json"))
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(entities_and_chunks, f, ensure_ascii=False, indent=4)

        logger.info("textbook %s processed", textbook)


def build_textbook_faiss_index():
    """
    每个 categoty 构建 faiss index 并保存到文件
    """
    faiss_index_dir = "./data/textbook/en_faiss_index/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                device=device)

    for item in settings.EN_TEXTBOOKS:
        category = item["category"]
        textbooks = item["file_path"]
        # 每个 category 收集所有 chunks
        chunks = []
        metadata = []

        for textbook in textbooks:
            logger.info("processing textbook: %s", textbook)

            with open(textbook, "r", encoding="utf-8") as f:
                text = f.read()

            # 分块
            current_textbook_chunks = TEXT_SPLITTER.split_text(text)
            chunks.extend(current_textbook_chunks)

            for chunk in current_textbook_chunks:
                metadata.append({
                    "chunk": chunk,
                    "source": textbook,
                })

        # 转化为 embeddings
        logger.info("encoding %d chunks", len(chunks))
        embeddings = model.encode(chunks,
                                  convert_to_numpy=True,
                                  show_progress_bar=True)
        # faiss 需要 float32 格式
        if embeddings.dtype != 'float32':
            embeddings = embeddings.astype('float32')

        # embeddings 归一化
        logger.info("normalizing embeddings")
        faiss.normalize_L2(embeddings)

        # 构建 faiss index
        logger.info("building faiss index")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        # 保存 faiss index
        logger.info("saving faiss index")
        faiss_index_file = os.path.join(faiss_index_dir, f"{category}.index")
        faiss.write_index(index, faiss_index_file)

        # 保存 metadata
        logger.info("saving metadata")
        metadata_file = os.path.join(faiss_index_dir,
                                     f"{category}_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)


def main():
    """
    主程序入口
    """
    # build_textbook_inverted_index()
    build_textbook_faiss_index()


if __name__ == "__main__":
    main()
