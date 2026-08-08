import sys
sys.path.append("..")
# logger_setup.py
import logging
import sys
import io
from logging.handlers import RotatingFileHandler

_CONFIGURED = False  # 防重复初始化


class _LoggerWriter(io.TextIOBase):
    """把 print 的输出写进 logging。"""
    def __init__(self, log_func):
        super().__init__()
        self._log = log_func

    def write(self, message):
        # logging 自带换行，这里去掉空白行，保留多行拆分
        if message and message.strip():
            for line in message.rstrip().splitlines():
                self._log(line)

    def flush(self):
        pass


def setup_logging(
    log_file: str = "run.log",
    level: int = logging.INFO,
    max_bytes: int = 20 * 1024 * 1024,  # 20MB
    backup_count: int = 3,
    also_stdout: bool = False,          # 需要同时在控制台打印可设 True
    encoding: str = "utf-8",
):
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [pid:%(process)d tid:%(threadName)s] "
            "%(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件滚动日志
    fh = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 可选：同时打印到终端（不需要的话就只写文件）
    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # 关键：把 print() 重定向到 logging
    sys.stdout = _LoggerWriter(logging.getLogger("STDOUT").info)
    sys.stderr = _LoggerWriter(logging.getLogger("STDERR").error)

    # 如果你还想捕获 warnings.warn：
    logging.captureWarnings(True)

    _CONFIGURED = True
