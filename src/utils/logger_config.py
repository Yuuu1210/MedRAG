"""
日志工具
"""
import logging

LOG_LEVEL = logging.INFO  # 可配置的日志级别
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = "./experiments/medrag_log.log"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# 创建一个格式化器
formatter = logging.Formatter(LOG_FORMAT)

# 创建一个控制台处理程序
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
ch.setFormatter(formatter)

# 获取根 logger 并进行配置
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
logger.addHandler(ch)

if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_logger(name):
    """
    返回一个指定名称的 logger
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    # 示例用法
    my_logger = get_logger(__name__)
    my_logger.debug("This is a debug message from logger_config")
