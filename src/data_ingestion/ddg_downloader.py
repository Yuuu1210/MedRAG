"""
根据 duckduckgo query 查询, 下载 pdf
"""
import json
import os
import re
import time

import requests
from duckduckgo_search import DDGS

import settings
from src.llm_clients import doubao_client
from src.utils import logger_config

logger = logger_config.get_logger(__name__)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36",
})

llm = doubao_client.DoubaoClient()


class DdgResult:
    """
    DuckDuckGo 搜索结果
    """
    title: str
    href: str
    body: str


def sanitize_filename(text, max_length=70):
    """
    (Helper) 根据文本生成一个安全的文件名
    """
    # 1. 移除非字母数字字符，但保留下划线和短横线
    text = re.sub(r'[^\w\s-]', '', text)
    # 2. 用下划线替换空格和多个连续的下划线/短横线
    text = re.sub(r'[-\s_]+', '_', text)
    # 3. 截断
    return text[:max_length].strip('_')


def download_pdf(pdf_link: str, file_path: str) -> bool:
    """
    (Helper) 下载 pdf 文件, 成功返回 True, 失败返回 False

    Args:
        pdf_link: pdf 链接
        file_path: 最终文件名

    Returns:
        bool: 下载是否成功
    """
    # 检查文件是否已存在
    if os.path.exists(file_path):
        logger.info("PDF exists: %s, skip download", file_path)
        return True

    try:
        # 执行下载
        response = session.get(
            pdf_link,
            stream=True,
            allow_redirects=True,
            timeout=30  # 添加超时设置
        )
        response.raise_for_status()

        # 流式写入文件
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤 keep-alive 空 chunk
                    f.write(chunk)

        logger.info("PDF downloaded: %s", file_path)
        return True

    except Exception as e:
        logger.error("Unexpected error: %s | URL: %s", str(e), pdf_link)
        return False


# def search_duckduckgo(keywords: str) -> list[DdgResult]:
#     """
#     (Helper) 使用 duckduckgo 搜索引擎搜索关键词

#     Args:
#         keywords: 搜索关键词

#     Returns:
#         list[DdgResult]: DuckDuckGo 搜索结果
#     """
#     keywords += " filetype:pdf"

#     headers = {
#         "User-Agent":
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
#     }
#     ddgs = DDGS(headers=headers)
#     time.sleep(1)
#     results = ddgs.text(keywords,
#                         region="us-en",
#                         max_results=settings.DDG_API_RETMAX)

#     return results


def search_serper(keywords: str) -> list:
    """
    (Helper) 使用 serper 搜索引擎搜索关键词

    Args:
        keywords: 搜索关键词
    
    Returns:
        list: 搜索结果
    """
    keywords += " filetype:pdf"

    url = "https://google.serper.dev/search?"
    params = {
        "q": keywords,
        "apiKey": settings.SERPER_API_KEY,
        "num": settings.DDG_API_RETMAX,
        'tbs': 'qdr:y5'  # 限制搜索结果为过去5年内的
    }
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
    }

    time.sleep(1)
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    results = []
    organic = response.json().get("organic", [])
    for item in organic:
        results.append({
            "title": item.get("title", ""),
            "href": item.get("link", ""),
            "body": item.get("snippet", "")
        })

    return results


def qa_download_pdfs_from_ddg() -> None:
    """
    在 duckduckgo 中搜索并下载pdf文件

    如果 202 Ratelimit, 
    尝试 pip install --upgrade --quiet duckduckgo-search
    或者更换梯子节点
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        query_id = item["id"]
        query = item["query"]
        option_texts = list(item["options"].values())
        ddg_queries = item["ddg_queries"]

        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        save_dir = f"./experiments/medmcqa/docs/{query_id}/"
        os.makedirs(save_dir, exist_ok=True)
        for i, ddg_query in enumerate(ddg_queries):
            logger.info("processing query %s option %d", query_id, i)
            attempts = 0
            while True:
                try:
                    # 使用 Serper 搜索引擎
                    results = search_serper(ddg_query)
                    break
                except requests.exceptions.RequestException as e:
                    attempts += 1
                    logger.error("attempts %d, Error during search: %s",
                                 attempts, str(e))
                    if attempts > 5:
                        return
                    time.sleep(3)

            for result in results:
                title = result["title"]
                pdf_link = result["href"]
                content = result["body"]

                if llm.qa_is_website_relevant(query, option_texts[i], title,
                                              content)["relevant"]:
                    filename = sanitize_filename(pdf_link.split("/")[-1])
                    pdf_file = os.path.join(save_dir, f"{filename}.pdf")
                    download_pdf(pdf_link, pdf_file)


def fc_download_pdfs_from_ddg() -> None:
    """
    fact checking 数据集使用, 在 duckduckgo 中搜索并下载pdf文件

    如果 202 Ratelimit, 
    尝试 pip install --upgrade --quiet duckduckgo-search
    或者更换梯子节点
    """
    query_file = "./experiments/healthfc/jsons/doubao_query.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 750

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        query_id = item["id"]
        query = item["query"]
        ddg_query = item["ddg_query"]
        save_dir = f"./experiments/healthfc/docs/{query_id}/"

        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("processing query %s", query_id)
        results = search_duckduckgo(ddg_query)

        for result in results:
            title = result["title"]
            pdf_link = result["href"]
            content = result["body"]

            if llm.fc_is_website_relevant(query, title, content)["relevant"]:
                download_pdf(pdf_link, save_dir)


def main():
    """
    主程序入口
    """
    qa_download_pdfs_from_ddg()
    # fc_download_pdfs_from_ddg()


if __name__ == "__main__":
    main()
