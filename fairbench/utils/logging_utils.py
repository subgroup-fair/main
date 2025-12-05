from __future__ import annotations
import logging, time, threading
from contextlib import contextmanager

try:
    import torch
except Exception:
    torch = None

def setup_logger(name: str = "fair", logfile: str | None = None, level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers: 
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
    t0 = time.time()
    log.info(f"▶ {label} ...")
    try:
        yield
    finally:
        log.info(f"✔ {label} done in {time.time() - t0:.2f}s")

def mem_str(device=None):
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
