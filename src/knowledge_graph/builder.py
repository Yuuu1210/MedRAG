"""
根据初步过滤后的文本构建知识图谱

neo4j 浏览器: http://localhost:7474/
"""
import json
import os

import numpy as np
from neo4j import GraphDatabase

import settings
from src.llm_clients import doubao_client
from src.utils import logger_config

llm = doubao_client.DoubaoChat()

logger = logger_config.get_logger(__name__)


def create_all_graphs():
    """
    为 [min_id, max_id] 范围内的所有问题创建知识图谱
    """
    final_score_dir = "./data/pubmed_and_ddg/medmcqa/inverted_index_final_score/"
    sim_matrix_dir = "./data/pubmed_and_ddg/medmcqa/similarity_matrix/"
    final_score_threshold = 0.5
    min_id = 7
    max_id = 7

    for query_id in range(min_id, max_id + 1):
        final_score_file = os.path.join(final_score_dir,
                                        f"dev_{query_id}_final_score.json")
        sim_matrix_file = os.path.join(sim_matrix_dir,
                                       f"dev_{query_id}_sim_matrix.npy")

        if not os.path.exists(final_score_file):
            logger.info("query_id %d final_score_file not exists", query_id)
            continue

        logger.info("processing query_id %d", query_id)
        with open(final_score_file, "r", encoding="utf-8") as f:
            final_score_data = json.load(f)

        chunks = []
        for item in final_score_data:
            if item["final_score"] < final_score_threshold:
                continue
            chunks.append(item)

        create_graph(query_id, chunks, np.load(sim_matrix_file))


def create_graph(
    query_id: int,
    chunks: list[dict[str, str]],
    sim_matrix: np.ndarray,
):
    """
    为单个问题创建知识图谱

    知识图谱的节点为 chunk
    根据 cosine similarity 连接边
    """
    sim_threshold = 0.7

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )

    with driver.session() as session:
        # 清空现有图形（可选，根据需求决定是否保留）
        session.run("MATCH (n) DETACH DELETE n")

        # 创建节点
        for i, chunk in enumerate(chunks):
            session.run("""
                CREATE (n:Sentence {
                    id: $id,
                    text: $text,
                    source: $source,
                    score: $score
                })
                """,
                        id=i,
                        text=chunk["sentence"],
                        source=os.path.basename(chunk["source"]).replace(
                            ".json", ""),
                        score=float(chunk["final_score"]))

        # 创建边（无向关系）
        n = len(chunks)
        for i in range(n):
            for j in range(i + 1, n):  # 避免重复和自环
                similarity = sim_matrix[i][j]
                if similarity > sim_threshold:
                    session.run("""
                        MATCH (a:Sentence {id: $id1}), (b:Sentence {id: $id2})
                        CREATE (a)-[r:SIMILAR_TO {
                            similarity: $similarity
                        }]->(b)
                        """,
                                id1=i,
                                id2=j,
                                similarity=float(similarity))

        session.run(f"""
            CALL apoc.export.cypher.all(
                'graph_{query_id}.cypher',
                {{format: 'cypher-shell'}}
            )
        """)
        logger.info("query_id %d graph exported", query_id)

    driver.close()


def import_graph_from_cypher(query_id):
    """
    从 cypher 文件导入 neo4j 图
    """
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )

    cypher_file = f"./data/neo4j_exports/graph_{query_id}.cypher"

    with driver.session() as session:
        # 清空现有图形
        session.run("MATCH (n) DETACH DELETE n")

        # 读取并执行 Cypher 文件
        with open(cypher_file, "r", encoding="utf-8") as f:
            cypher_commands = f.read()
            # 可能需要分割命令（如果文件包含多个语句）
            for cmd in cypher_commands.split(";"):
                if cmd.strip():
                    session.run(cmd)

    driver.close()


def main():
    """
    主程序入口
    """
    pass


if __name__ == "__main__":
    main()
