"""
使用 MinerU 解析 pdf

-----------------------------------------------------
AutoDL vpn: source /etc/network_turbo
取消加速: unset http_proxy && unset https_proxy
终端重新加载配置: source ~/.bashrc
清理回收站: rm -rf 文件夹/.Trash-0
-----------------------------------------------------
"""
import json
import os
import warnings

import torch
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import (FileBasedDataReader,
                                               FileBasedDataWriter)
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

from src.utils import logger_config

warnings.simplefilter("ignore", category=FutureWarning)

logger = logger_config.get_logger(__name__)


def check_cudnn_version():
    """
    查看 cudnn 版本
    """
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("", torch.backends.cudnn.version())


def modify_magic_pdf_json():
    """
    修改 magic-pdf.json 文件
    """
    filename = "magic-pdf.json"

    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # 进行修改操作
    data["device-mode"] = "cuda"

    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def mineru_process_pdf(pdf_path: str):
    """
    使用 mineru 解析 pdf
    """
    # args
    name_without_suff = os.path.splitext(os.path.basename(pdf_path))[0]

    # prepare env
    local_md_dir = os.path.join(os.path.dirname(pdf_path), name_without_suff)
    local_image_dir = os.path.join(local_md_dir, "images")
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(
        local_image_dir), FileBasedDataWriter(local_md_dir)

    # read bytes
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_path)  # read the pdf content

    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    ## inference
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)

        ## pipeline
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

    else:
        infer_result = ds.apply(doc_analyze, ocr=False)

        ## pipeline
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    ### dump content list
    pipe_result.dump_content_list(md_writer,
                                  f"{name_without_suff}_content_list.json",
                                  image_dir)


def pdf_to_markdown():
    """
    将 pdf_dir 目录下的所有 pdf 调用 magic-pdf 转为 markdown。
    """
    min_id = 1
    max_id = 450

    for query_id in range(min_id, max_id + 1):
        pdf_dir = f"./experiments/medmcqa/docs/{query_id}/"

        if not os.path.exists(pdf_dir):
            logger.info("%s not exists, skip...", pdf_dir)
            continue

        logger.info("processing query %d", query_id)
        for filename in os.listdir(pdf_dir):
            # 忽略大小写检查 .pdf 结尾
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(pdf_dir, filename)

            # MinerU 处理 pdf
            try:
                mineru_process_pdf(pdf_path)
                logger.info("pdf %s processed", filename)
            except Exception as e:
                logger.error("Error processing %s", pdf_path)


def main():
    """
    主程序入口
    """
    pdf_to_markdown()


if __name__ == "__main__":
    main()
