"""
根据 pubmed query, 在 pubmed 中下载论文
"""
import json
import os
import shutil
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter

import settings
from src.llm_clients import doubao_client
from src.utils import logger_config

llm = doubao_client.DoubaoClient()
logger = logger_config.get_logger(__name__)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/pdf",
})


def fetch_pmids_by_pubmed_query():
    """
    通过 pubmed query 获取 pmid
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    pmid_file = "./experiments/medmcqa/jsons/doubao_pmid.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 961
    max_id = 1000

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    if os.path.exists(pmid_file):
        with open(pmid_file, "r", encoding="utf-8") as f:
            pmid_data = json.load(f)

        # 键是 query_id，值是包含 id/query/pmids 的字典
        id_and_info = {item["id"]: item for item in pmid_data}
    else:
        id_and_info = {}

    for item in query_data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        query = item["query"]
        options = item["options"]
        pubmed_queries = item["pubmed_queries"]
        logger.info("processing query %s", query_id)

        for pubmed_query in pubmed_queries:
            pubmed_query = " AND ".join(pubmed_query)
            pmids = search_pubmed_by_query(pubmed_query)

            # 初始化或更新条目
            if query_id not in id_and_info:
                id_and_info[query_id] = {
                    "id": query_id,
                    "query": query,
                    "options": options,
                    "pmids": []
                }
            id_and_info[query_id]["pmids"].extend(pmids)
            id_and_info[query_id]["pmids"] = list(
                set(id_and_info[query_id]["pmids"]))  # 原地去重

            # 最终转换为 List[dict]
            result_list = list(id_and_info.values())

            with open(pmid_file, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)


def fc_fetch_pmids_by_pubmed_query():
    """
    通过 pubmed query 获取 pmid
    """
    query_file = "./data/dataset/healthfc/doubao_query.json"
    pmid_file = "./data/dataset/healthfc/doubao_pmid.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 453
    max_id = 750

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    with open(pmid_file, "r", encoding="utf-8") as f:
        pmid_data = json.load(f)

    # 键是 query_id，值是包含 id/query/pmids 的字典
    id_and_info = {item["id"]: item for item in pmid_data}

    for item in query_data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        query = item["question"]
        logger.info("processing query %s", query_id)
        pubmed_query = " AND ".join(item["pubmed_query"])
        pmids = search_pubmed_by_query(pubmed_query)

        # 初始化或更新条目
        if query_id not in id_and_info:
            id_and_info[query_id] = {
                "id": query_id,
                "question": query,
                "pmids": []
            }
        id_and_info[query_id]["pmids"].extend(pmids)
        id_and_info[query_id]["pmids"] = list(
            set(id_and_info[query_id]["pmids"]))  # 原地去重

        # 最终转换为 list[dict]
        result_list = list(id_and_info.values())

        with open(pmid_file, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)


def search_pubmed_by_query(query: str) -> list[str]:
    """
    (Helper) 根据 query 在 pubmed 中搜索并返回 pmids
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    query += " AND medline[sb] AND pubmed pmc open access[filter]"

    # esearch参数, more info: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": settings.PUBMED_API_RETMAX,
        "retmode": "json",
        "sort": "relevance",  # 按相关性排序
        "datetype": "pdat",  # 指定日期类型为发表日期
        "reldate": 5 * 365,  # 限制为最近 N 天内发表
        "api_key": settings.NCBI_API_KEY,
        "email": settings.NCBI_EMAIL,
    }

    time.sleep(0.5)  # 避免请求过于频繁
    response = session.get(esearch_url, params=params)
    results = response.json()

    # 返回pmid列表
    try:
        return results["esearchresult"]["idlist"]
    except KeyError as e:
        logger.error("error: %s", e)
        logger.info("response content: %s", results)
        return []


def get_pubmed_metadata():
    """
    根据 pmid 获取 PubMed 元数据 (PMCID/DOI/标题/摘要)
    """
    pmid_file = "./experiments/medmcqa/jsons/doubao_pmid.json"
    metadata_file = "./experiments/medmcqa/jsons/pubmed_metadata.json"
    # pmid_file = "./data/dataset/healthfc/doubao_pmid.json"
    # metadata_file = "./data/dataset/healthfc/pubmed_metadata.json"
    min_id = 1
    max_id = 1000

    # 初始化元数据存储结构
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {}

    # 配置HTTP适配器（自动重试）
    session.mount('https://', HTTPAdapter(max_retries=3))

    with open(pmid_file, "r", encoding="utf-8") as f:
        pmid_data = json.load(f)

    for item in pmid_data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        pmids = item["pmids"]
        # 跳过空PMID列表和非列表类型（确保数据格式）
        if not isinstance(pmids, list) or len(pmids) == 0:
            logger.warning("Invalid PMIDs for query %s: %s", query_id,
                           str(pmids))
            continue

        logger.info("Processing query %s with %d PMIDs", query_id, len(pmids))

        params = {
            'db': 'pubmed',
            'id': ','.join(map(str, pmids)),  # 确保PMID为字符串
            'retmode': 'xml',
            "api_key": settings.NCBI_API_KEY,
            "email": settings.NCBI_EMAIL,
        }

        try:
            # 精确控制请求间隔（符合NCBI API政策）
            time.sleep(0.5)
            response = session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
                timeout=(3, 30))
            response.raise_for_status()

            # 解析并更新元数据
            result = parse_xml_get_metadata(response.content, result)
            # 实时保存结果（每次请求后保存）
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            logger.error("Query %s failed: %s", query_id, str(e))
            continue


def parse_xml_get_metadata(xml_content: bytes, result: dict) -> dict:
    """
    解析 PubMed XML 数据
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error("XML parsing failed: %s", str(e))
        return result

    for article in root.findall('.//PubmedArticle'):
        # PMID 提取
        pmid_elem = article.find('.//MedlineCitation/PMID')
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text.strip()

        # 初始化记录
        current_info = result.get(pmid, {
            'doi': None,
            'pmcid': None,
            'title': None,
            'abstract': None
        })

        # 提取标题（处理可能的内联标签）
        title_elem = article.find('.//ArticleTitle')
        if title_elem is not None:
            # 合并所有文本内容（包括子元素文本）
            current_info['title'] = ''.join(title_elem.itertext()).strip()

        # 提取摘要（结构化处理）
        abstract_elem = article.find('.//Abstract')
        if abstract_elem is not None:
            abstract_parts = []
            for text_elem in abstract_elem.findall('.//AbstractText'):
                # 处理带标签的段落
                label = text_elem.get('Label')
                text = ''.join(text_elem.itertext()).strip()

                if label:
                    # 检查标签是否已包含在文本中
                    if not text.lower().startswith(label.lower()):
                        text = f"{label}: {text}"
                abstract_parts.append(text)

            current_info['abstract'] = '\n\n'.join(
                abstract_parts) if abstract_parts else None

        # 提取标识符（保持原有逻辑）
        pubmed_data = article.find('.//PubmedData')
        if pubmed_data is not None:
            id_list = pubmed_data.find('.//ArticleIdList')
            if id_list is not None:
                for id_elem in id_list.findall('ArticleId'):
                    id_type = id_elem.get('IdType', '').lower()
                    text = id_elem.text.strip()

                    if id_type == 'doi' and not current_info['doi']:
                        current_info['doi'] = text
                    elif id_type == 'pmc':
                        pmcid = text.upper()
                        if not pmcid.startswith('PMC'):
                            pmcid = f"PMC{pmcid}"
                        current_info['pmcid'] = pmcid

        # 备选PMCID提取
        if not current_info['pmcid']:
            article_elem = article.find('.//Article')
            if article_elem is not None:
                pmc_elem = article_elem.find('.//ArticleId[@IdType="pmc"]')
                if pmc_elem is not None and pmc_elem.text:
                    pmcid = pmc_elem.text.strip().upper()
                    current_info[
                        'pmcid'] = f"PMC{pmcid}" if not pmcid.startswith(
                            'PMC') else pmcid

        result[pmid] = current_info

    return result


def qa_download_pdfs_from_pubmed():
    """
    (qa dataset) 下载 pdf
    """
    pmid_file = "./experiments/medmcqa/jsons/doubao_pmid.json"
    metadata_file = "./experiments/medmcqa/jsons/pubmed_metadata.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000

    with open(pmid_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for item in data:
        query_id = item["id"]
        query = item["query"]
        options = item["options"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("processing query %s", query_id)

        for pmid in item["pmids"]:
            if pmid not in metadata:
                continue

            title = metadata[pmid]["title"]
            abstract = metadata[pmid]["abstract"]
            pmcid = metadata[pmid]["pmcid"]

            if not llm.qa_is_paper_relevant(query, options, title,
                                            abstract)["relevant"]:
                continue

            logger.info("downloading pdf for %s", pmid)
            save_dir = f"./experiments/medmcqa/docs/{query_id}"
            pdf_file = f"{save_dir}/PMID{pmid}.pdf"
            # 确保 save_dir 存在
            os.makedirs(save_dir, exist_ok=True)
            if pmcid:
                download_pdf_by_pmcid(pdf_file, pmcid)


def fc_download_pdfs_from_pubmed():
    """
    fact checking 数据集下载 pdf
    """
    pmid_file = "./data/dataset/healthfc/doubao_pmid.json"
    metadata_file = "./data/dataset/healthfc/pubmed_metadata.json"
    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 101
    max_id = 750

    with open(pmid_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        query_id = item["id"]
        query = item["question"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("processing query %s", query_id)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for pmid in item["pmids"]:
            if pmid not in metadata:
                continue

            title = metadata[pmid]["title"]
            abstract = metadata[pmid]["abstract"]
            pmcid = metadata[pmid]["pmcid"]
            doi = metadata[pmid]["doi"]

            if not llm.fc_is_paper_relevant(query, title, abstract)["relevant"]:
                continue

            logger.info("downloading pdf for %s", pmid)
            save_dir = f"./experiments/medmcqa/docs/{query_id}"
            pdf_file = f"{save_dir}/PMID{pmid}.pdf"
            # 确保 save_dir 存在
            os.makedirs(save_dir, exist_ok=True)
            if pmcid:
                download_pdf_by_pmcid(pdf_file, pmcid)


def download_pdf_by_pmcid(pdf_file: str, pmcid: str):
    """
    尝试通过 pmcid 下载论文 pdf
    """
    filename_with_ext = os.path.basename(pdf_file)
    pmid = filename_with_ext[len("PMID"):-len(".pdf")]

    # 检查pdf是否已存在
    if os.path.exists(pdf_file):
        logger.info("skip download, pdf exists for %s", pmid)
        return

    # 若pdf不存在，尝试下载PDF
    fcgi_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    params_fcgi = {
        'id': pmcid,
    }
    response_fcgi = session.get(fcgi_url,
                                params=params_fcgi,
                                stream=True,
                                timeout=60,
                                allow_redirects=True)
    response_fcgi.raise_for_status()

    xml_content = response_fcgi.text
    root = ET.fromstring(xml_content)
    tgz_link_node = root.find(".//link[@format='tgz']")
    if tgz_link_node is None:
        logger.error("未找到 tgz 链接节点，可能是 PMC ID 无效或没有可用的 PDF。")
        return
    tgz_url = tgz_link_node.attrib['href']

    # 修正 FTP URL 为 HTTPS
    if tgz_url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        tgz_url = tgz_url.replace("ftp://ftp.ncbi.nlm.nih.gov/",
                                  "https://ftp.ncbi.nlm.nih.gov/", 1)

    # 创建一个临时目录来处理 .tar.gz 文件
    with tempfile.TemporaryDirectory(prefix=f"PMID{pmid}_") as tmp_dir_path:
        tgz_filename = os.path.join(tmp_dir_path, f"{pmid}.tar.gz")

        # 1. 下载 .tar.gz 文件
        try:
            response_tgz = session.get(tgz_url, stream=True, timeout=180)
            response_tgz.raise_for_status()
            with open(tgz_filename, 'wb') as f_tgz:
                for chunk in response_tgz.iter_content(chunk_size=8192 *
                                                       4):  # 增加 chunk size
                    f_tgz.write(chunk)
            print(f"成功下载 .tar.gz 文件。")
        except requests.exceptions.RequestException as e_tgz_dl:
            print(f"错误: 下载 .tar.gz 文件失败: {e_tgz_dl}")
            return False

        # 2. 解压 .tar.gz 文件
        with tarfile.open(tgz_filename, "r:gz") as tar:
            members = tar.getmembers()
            pdf_files_in_tar = []
            for member in members:
                if member.isfile() and member.name.lower().endswith('.pdf'):
                    pdf_files_in_tar.append(member)

            if not pdf_files_in_tar:
                print(f"错误: 在 .tar.gz 文件中未找到 PDF 文件。")
                return False

            pdf_member_to_extract = pdf_files_in_tar[0]
            if len(pdf_files_in_tar) > 1:
                print(
                    f"警告: .tar.gz 中找到多个 PDF 文件 ({[m.name for m in pdf_files_in_tar]})，将尝试提取最合适的。"
                )
                preferred_name_patterns = ['article']

                for pf_member in pdf_files_in_tar:
                    if any(pattern in pf_member.name.lower()
                           for pattern in preferred_name_patterns):
                        pdf_member_to_extract = pf_member
                        break

            tar.extract(pdf_member_to_extract, path=tmp_dir_path)
            # 解压时，tar.extract 会保留原始的目录结构（如果 tar 包内有的话）
            # 所以 extracted_pdf_path_in_tmp 需要是解压后文件在 tmp_dir_path 下的完整路径
            extracted_pdf_path_in_tmp = os.path.join(tmp_dir_path,
                                                     pdf_member_to_extract.name)

            if os.path.exists(extracted_pdf_path_in_tmp):
                print(f"成功解压 PDF: {extracted_pdf_path_in_tmp}")
                shutil.move(extracted_pdf_path_in_tmp, pdf_file)
                print(f"PDF 文件已移动到: {pdf_file}")
                return True
            else:
                print(f"错误: 解压后未找到预期的 PDF 文件路径 ({extracted_pdf_path_in_tmp})。")
                return False


def main():
    """
    主程序入口
    """
    # fetch_pmids_by_pubmed_query()
    # fc_fetch_pmids_by_pubmed_query()
    # get_pubmed_metadata()
    qa_download_pdfs_from_pubmed()
    # fc_download_pdfs_from_pubmed()


if __name__ == "__main__":
    main()
