"""
统计 pdf 相关信息
"""
import json
import os
import re

import fitz


def is_pdf_valid(pdf_path: str) -> bool:
    """
    判断 pdf 是否有效
    """
    doc = None
    try:
        # 使用 'with' 语句确保文件资源被正确关闭
        with fitz.open(pdf_path) as doc:
            # 1. 基本打开检查：如果 'with' 语句成功执行，表示文件至少可以被识别和打开。

            # 2. 页面数量检查：获取页面数。如果失败，可能意味着内部结构问题。
            num_pages = doc.page_count
            print(f"'{pdf_path}' 成功打开，包含 {num_pages} 页。")

            if num_pages == 0:
                print(f"'{pdf_path}' 包含 0 页。这可能是有效的，但也可能表明有问题。")

        return True

    except Exception as e:
        # 捕获其他可能的异常（如内存错误、权限问题等）
        print(f"处理 '{pdf_path}' 时发生意外错误: {e}")
        return False


def is_pubmed_pdf(pdf_name: str):
    """
    根据 pdf 文件名判断 pdf 是否来自 pubmed
    """
    pattern = r"^PMID\d{8}\.pdf$"
    match = re.fullmatch(pattern, pdf_name, re.IGNORECASE)
    return bool(match)


def count_pdf_statistics():
    """
    统计 pdf 相关信息
    """
    min_id = 1
    max_id = 1000

    damaged_pubmed_pdfs = []
    valid_pubmed_pdfs = []
    damaged_ddg_pdfs = []
    valid_ddg_pdfs = []
    unknown_pdfs = []
    for query_id in range(min_id, max_id + 1):
        pdf_dir = f"./experiments/medmcqa/docs/{query_id}/"

        if not os.path.exists(pdf_dir):
            print(f"{pdf_dir} not exists")
            continue

        for filename in os.listdir(pdf_dir):
            full_path = os.path.join(pdf_dir, filename)

            # 检查是否是文件，并且后缀是 .pdf 或 .PDF
            if not os.path.isfile(full_path):
                continue
            if not filename.endswith(('.pdf', '.PDF')):
                continue

            if is_pdf_valid(full_path) and is_pubmed_pdf(filename):
                valid_pubmed_pdfs.append(full_path)
            elif not is_pdf_valid(full_path) and is_pubmed_pdf(filename):
                damaged_pubmed_pdfs.append(full_path)
            elif is_pdf_valid(full_path) and not is_pubmed_pdf(filename):
                valid_ddg_pdfs.append(full_path)
            elif not is_pdf_valid(full_path) and not is_pubmed_pdf(filename):
                damaged_ddg_pdfs.append(full_path)
            else:
                unknown_pdfs.append(full_path)

    print("valid_pubmed_pdfs:", len(valid_pubmed_pdfs))
    print("damaged_pubmed_pdfs:", len(damaged_pubmed_pdfs))
    print("valid_ddg_pdfs:", len(valid_ddg_pdfs))
    print("damaged_ddg_pdfs:", len(damaged_ddg_pdfs))
    print("unknown_pdfs:", len(unknown_pdfs))


def main():
    """
    主程序入口
    """
    count_pdf_statistics()


if __name__ == "__main__":
    main()
