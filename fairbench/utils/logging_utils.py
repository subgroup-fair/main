# fairbench/utils/logging_utils.py
from __future__ import annotations
import logging, time, threading
from contextlib import contextmanager

try:
    import torch
except Exception:
    torch = None

def setup_logger(name: str = "fair", logfile: str | None = None, level=logging.INFO):
    """콘솔(+선택적으로 파일) 로거 셋업."""
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:  # 중복 핸들러 방지
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(fmt)
            log.addHandler(fh)
    return log

@contextmanager
def Timer(log, label: str):
    """with Timer(log, 'label') 로 구간 시간 측정."""
    t0 = time.time()
    log.info(f"▶ {label} ...")
    try:
        yield
    finally:
        log.info(f"✔ {label} done in {time.time() - t0:.2f}s")

def mem_str(device=None):
    """CUDA 메모리 사용량 문자열."""
    if torch is None:
        return "CPU"
    try:
        if device is not None and hasattr(device, "type") and device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            return f"{alloc:.0f}MB/{reserv:.0f}MB"
    except Exception:
        pass
    return "CPU"

class Heartbeat:
    """N초마다 '살아있다' 로그."""
    def __init__(self, log, secs: int):
        self.log, self.secs = log, secs
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.secs > 0:
            self._thr.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.secs):
            self.log.info("[hb] still training...")
