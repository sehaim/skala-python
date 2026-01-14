# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : 구조적 로깅 및 컨텍스트 추적 실습
# 작성일 : 2026-01-14
# 기능 : 여러 프로세스가 동시에 로그를 남겨도 한 파일에 안전하게 기록(단일 Writer) + JSON Logging 포맷 적용
# 실습 포인트 :
# - Race Condition : 여러 프로세스가 하나의 파일에 동시에 write 하면 로그가 섞이거나 깨질 수 있음을 이해
# - Queue 기반 Logging : 워커는 Queue로만 로그를 보내고, 메인 프로세스가 단일 Listener로 파일에 기록하여 충돌 방지
# - JSON Logging : ELK, Splunk 등 로그 분석 도구에서 파싱하기 쉽도록 JSON Lines(JSONL) 형태로 표준화
# - 식별자 포함 : ProcessID(어떤 프로세스가 남겼는지), TaskID(작업 고유 ID)를 모든 로그에 포함
# ------------------------------------------------------------

from __future__ import annotations

import csv
import json
import os
import time
import uuid
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import logging.handlers


# =========================
# 1. 커스텀 JSON 포매터
# =========================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # 테스트 로그(application_logs.json) 포맷에 맞춤:
        # timestamp, level, batch_id, task_id, process_id, thread_id, stage, message, context, (optional) exception
        level = record.levelname
        if level == "WARNING":
            level = "WARN"

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": level,
            "batch_id": getattr(record, "batch_id", None),
            "task_id": getattr(record, "task_id", None),
            "process_id": record.process,
            "thread_id": getattr(record, "thread_id", None),
            "stage": getattr(record, "stage", None),
            "message": record.getMessage(),
            "context": getattr(record, "context", None) or {},
        }

        # 예외는 테스트 로그처럼 exception 객체로 별도 필드 구성
        exc = getattr(record, "exception", None)
        if exc:
            payload["exception"] = exc

        return json.dumps(payload, ensure_ascii=False)


# =========================
# 2. 워커 로거 / 리스너 구성
# =========================
def setup_worker_logger(
    log_queue: mp.Queue, level: int = logging.INFO
) -> logging.LoggerAdapter:
    logger = logging.getLogger("mp-worker")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(level)
    logger.addHandler(qh)

    # extra 값들은 worker_job에서 세팅/갱신
    return logging.LoggerAdapter(
        logger,
        extra={
            "batch_id": None,
            "task_id": None,
            "thread_id": None,
            "stage": None,
            "context": {},
            "exception": None,
        },
    )


def start_log_listener(
    log_queue: mp.Queue, log_path: str, level: int = logging.INFO
) -> logging.handlers.QueueListener:
    """
    메인 프로세스(단일 Writer)에서만 파일 핸들러를 열어 기록
    - 여러 프로세스가 동시에 파일에 쓰는 문제(Race Condition)를 구조적으로 제거
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter())

    listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        respect_handler_level=True,
    )
    listener.start()
    return listener


# =========================
# 3. 멀티프로세스 작업
# =========================
def _log(
    adapter: logging.LoggerAdapter,
    level: int,
    *,
    stage: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Dict[str, Any]] = None,
) -> None:
    adapter.extra["stage"] = stage
    adapter.extra["context"] = context or {}
    adapter.extra["exception"] = exception

    adapter.log(level, message)


def simulate_pipeline(adapter: logging.LoggerAdapter, task: Dict[str, Any]) -> None:
    batch_id = task["batch_id"]
    task_id = task["task_id"]
    job_type = task["job_type"]
    input_size_mb = int(task["input_size_mb"])

    adapter.extra["batch_id"] = batch_id
    adapter.extra["task_id"] = task_id

    # LOAD
    _log(
        adapter,
        logging.INFO,
        stage="LOAD",
        message="Task started",
        context={"job_type": job_type, "input_size_mb": input_size_mb},
    )
    time.sleep(0.01)

    # PREPROCESS
    _log(
        adapter,
        logging.INFO,
        stage="PREPROCESS",
        message="Data normalization completed",
        context={"elapsed_ms": 100 + (input_size_mb % 50)},
    )
    time.sleep(0.01)

    if job_type == "DATA_CLEANING":
        # CLEANING
        _log(
            adapter,
            logging.INFO,
            stage="CLEANING",
            message="Null values removed",
            context={"rows_affected": 10000 + (input_size_mb * 3)},
        )
        time.sleep(0.01)
        _log(
            adapter,
            logging.INFO,
            stage="CLEANING",
            message="Outliers filtered",
            context={"rows_removed": max(1, input_size_mb // 2)},
        )
        time.sleep(0.01)
        _log(
            adapter,
            logging.INFO,
            stage="FINISH",
            message="Cleaning completed",
            context={"output_rows": 10000 + (input_size_mb * 3) - max(1, input_size_mb // 2)},
        )
        return

    if job_type == "IMAGE_PROCESSING":
        _log(
            adapter,
            logging.INFO,
            stage="FINISH",
            message="Image processing completed",
            context={"images_processed": max(1, input_size_mb // 5)},
        )
        return

    # MODEL_TRAINING
    if job_type == "MODEL_TRAINING":
        if input_size_mb >= 1000:
            _log(
                adapter,
                logging.WARNING,
                stage="TRAINING",
                message="GPU contention detected",
                context={"gpu_id": 0, "lock_wait_ms": 300 + (input_size_mb % 200)},
            )
            time.sleep(0.01)
            _log(
                adapter,
                logging.ERROR,
                stage="TRAINING",
                message="CUDA out of memory",
                context={"retry_count": 1},
                exception={
                    "type": "OutOfMemoryError",
                    "stacktrace": "trainer.py:188 -> allocate_tensor()",
                },
            )
            time.sleep(0.01)
            _log(
                adapter,
                logging.INFO,
                stage="RETRY",
                message="Retrying training with reduced batch size",
                context={"new_batch_size": 16},
            )
            time.sleep(0.01)

        # 정상 진행 로그 (epoch 완료)
        _log(
            adapter,
            logging.INFO,
            stage="TRAINING",
            message="Epoch 3 completed",
            context={"loss": round(0.2 + (input_size_mb % 10) * 0.001, 3), "accuracy": 0.9},
        )
        time.sleep(0.01)
        _log(
            adapter,
            logging.INFO,
            stage="FINISH",
            message="Model training completed",
            context={"total_epochs": 10, "total_time_sec": 100 + (input_size_mb % 500)},
        )
        return


def worker_job(worker_idx: int, tasks: List[Dict[str, Any]], log_queue: mp.Queue) -> None:
    adapter = setup_worker_logger(log_queue)
    adapter.extra["thread_id"] = f"worker-{worker_idx}"

    for task in tasks:
        simulate_pipeline(adapter, task)


def load_tasks_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def split_round_robin(tasks: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        buckets[i % n].append(t)
    return buckets


def run_multiprocess_logging_demo(
    workers: int = max(1, (os.cpu_count() or 2) - 1),
    log_path: str = "./logs/multiprocess.jsonl",
) -> str:
    # 테스트 입력 파일 경로(업로드된 파일 기준)
    input_csv = "./data/input_tasks.csv"

    tasks = load_tasks_from_csv(input_csv)

    ctx = mp.get_context("spawn")
    log_queue: mp.Queue = ctx.Queue(maxsize=10000)

    listener = start_log_listener(log_queue, log_path)

    start = time.time()
    try:
        task_splits = split_round_robin(tasks, workers)

        procs: List[mp.Process] = []
        for i in range(workers):
            p = ctx.Process(target=worker_job, args=(i, task_splits[i], log_queue))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    finally:
        listener.stop()

    elapsed = round(time.time() - start, 4)

    # 메인 종료 로그(파일에는 남기지 않고 화면에만 출력)
    batch_id = tasks[0]["batch_id"] if tasks else "UNKNOWN_BATCH"
    demo_task_id = uuid.uuid4().hex

    print(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": "demo_finished",
                "batch_id": batch_id,
                "task_id": demo_task_id,
                "workers": workers,
                "log_path": log_path,
                "elapsed_sec": elapsed,
                "input_csv": input_csv,
                "total_tasks": len(tasks),
            },
            ensure_ascii=False,
        )
    )

    return demo_task_id


# =========================
# 4. 실행
# =========================
if __name__ == "__main__":
    task_id = run_multiprocess_logging_demo(
        workers=4,
        log_path="./output/application_logs.jsonl",
    )
