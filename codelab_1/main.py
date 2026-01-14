# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : 병렬 데이터 마스킹 및 텍스트 정규화 실습
# 기능 : 사용자 리뷰 데이터에 포함된 개인정보 마스킹 / 특정 금칙어를 표준 표현으로 순화하는 배치 프로세서 구축
# 실습 포인트 :
# - 정규 표현식(Regex)의 오버헤드 : 복잡한 정규표현식을 멀티 프로세스로 분산했을 때의 성능 이득 확인
# - 데이터 청킹(Chunking) : 대용량 데이터를 chunk로 나눠 병렬 처리할 때 처리량(Throughput) 비교
# - IPC 비용 : chunk가 너무 작으면 프로세스 간 통신(IPC) 비용이 커져 성능이 나빠짐을 관찰
# 작성일 : 2026-01-13
# ------------------------------------------------------------

from __future__ import annotations

import csv
import os
import re
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple


# 데이터 파일 경로 설정
CSV_PATH = "./data/reviews_500k.csv"


@dataclass(frozen=True)
class JobConfig:
    input_path: str
    output_path: str
    file_type: str
    text_field: str
    processes: int  # 0이면 단일 처리
    chunk_size: int
    max_rows: Optional[int] = None


# ============================================================
# 1. Regex 패턴(미리 컴파일) + 정규화 규칙
# ============================================================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?:"
    r"(?:\+?82[-.\s]?)?0?1[016789]"
    r"(?:[-.\s]?\d{3,4})"
    r"(?:[-.\s]?\d{4})"
    r")"
)
MULTI_SPACE_RE = re.compile(r"\s+")

NORMALIZE_MAP = {
    "갠춘": "괜찮음",
    "존맛": "아주 맛있음",
    "맛잇": "맛있",
    "별로임": "별로",
    "ㅈㅁㅌ": "아주 맛있음",
}


def mask_and_normalize(text: str) -> str:
    if not text:
        return ""

    text = EMAIL_RE.sub("****", text)
    text = PHONE_RE.sub("****", text)

    for src, dst in NORMALIZE_MAP.items():
        text = text.replace(src, dst)

    text = MULTI_SPACE_RE.sub(" ", text.strip())
    return text


# ============================================================
# 2. 입력 로더: CSV 읽기 (streaming)
# ============================================================
def iter_csv_rows(
    path: str, max_rows: Optional[int]
) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    f = open(path, "r", encoding="utf-8", newline="")
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []

    def row_iter():
        count = 0
        for row in reader:
            yield row
            count += 1
            if max_rows is not None and count >= max_rows:
                break
        f.close()

    return fieldnames, row_iter()


# ============================================================
# 3. Chunking + 처리 함수
# ============================================================
def chunked(
    iterable: Iterable[Dict[str, str]], size: int
) -> Iterable[List[Dict[str, str]]]:
    """iterable을 size개씩 묶어 리스트로 반환"""
    buf: List[Dict[str, str]] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def process_csv_chunk(
    rows: List[Dict[str, str]], text_field: str
) -> List[Dict[str, str]]:
    """chunk 내 각 행의 text_field를 clean_text로 변환"""
    for row in rows:
        raw = row.get(text_field, "") or ""
        row["clean_text"] = mask_and_normalize(raw)
    return rows


def mp_csv_worker(args):
    rows, text_field = args
    return process_csv_chunk(rows, text_field)


# ============================================================
# 4. 실행 파이프라인(단일 처리 vs 멀티 프로세스) + Throughput 측정
# ============================================================
def run_job(cfg: JobConfig) -> None:
    start = time.perf_counter()
    total_rows = 0

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    in_fieldnames, row_iter = iter_csv_rows(cfg.input_path, cfg.max_rows)
    text_field = cfg.text_field

    # field name을 clean_text로 교체
    out_fieldnames: List[str] = []
    for f in in_fieldnames:
        if f == text_field:
            out_fieldnames.append("clean_text")
        else:
            out_fieldnames.append(f)

    out_f = open(cfg.output_path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
    writer.writeheader()

    chunks = chunked(row_iter, cfg.chunk_size)

    if cfg.processes == 0:
        # 단일 프로세스: chunk 순회하며 처리
        for ch in chunks:
            processed = process_csv_chunk(ch, text_field)
            for row in processed:
                out_row = {k: v for k, v in row.items() if k != text_field}
                out_row.pop("clean_text", None)
                out_row["clean_text"] = row["clean_text"]
                writer.writerow(out_row)
            total_rows += len(processed)

    else:
        # 멀티 프로세스: chunk를 워커에 분산(IPC 단위 = chunk)
        with mp.Pool(processes=cfg.processes) as pool:
            for processed in pool.imap_unordered(
                mp_csv_worker,
                ((ch, text_field) for ch in chunks),
                chunksize=1,
            ):
                for row in processed:
                    out_row = {k: v for k, v in row.items() if k != text_field}
                    out_row.pop("clean_text", None)
                    out_row["clean_text"] = row["clean_text"]
                    writer.writerow(out_row)
                total_rows += len(processed)

    out_f.close()

    elapsed = time.perf_counter() - start
    rps = (total_rows / elapsed) if elapsed > 0 else 0.0

    mode = "single" if cfg.processes == 0 else f"multi({cfg.processes})"
    print(
        f"[{mode}] chunk_size={cfg.chunk_size:,}  time={elapsed:.2f}s  throughput={rps:,.0f} rows/s"
    )

    return elapsed, rps, total_rows


if __name__ == "__main__":
    chunk_sizes = [1_000, 5_000, 20_000]
    procs = os.cpu_count() or 4

    summary = []  # (chunk_size, single_time, single_rps, multi_time, multi_rps)

    for cs in chunk_sizes:
        print(f'\n{"-" * 26} chunk_size = {cs:,} {"-" * 26}')

        single_time, single_rps, _ = run_job(
            JobConfig(
                input_path=CSV_PATH,
                output_path=f"./out/reviews_500k_single_cs{cs}.csv",
                file_type="csv",
                text_field="review_text",
                processes=0,
                chunk_size=cs,
            )
        )

        multi_time, multi_rps, _ = run_job(
            JobConfig(
                input_path=CSV_PATH,
                output_path=f"./out/reviews_500k_multi_cs{cs}.csv",
                file_type="csv",
                text_field="review_text",
                processes=procs,
                chunk_size=cs,
            )
        )

        summary.append((cs, single_time, single_rps, multi_time, multi_rps))

    print("\n" + "=" * 30 + " chunk size에 따른 성능 비교 " + "=" * 30)
    print(
        "chunk_size | single_time(s) | single_rps | multi_time(s) | multi_rps | speedup(multi/single)"
    )
    print("-" * 95)

    for cs, st, sr, mt, mr in summary:
        speedup = (st / mt) if mt > 0 else 0.0
        print(
            f"{cs:>9,} | {st:>13.2f} | {sr:>10,.0f} | {mt:>12.2f} | {mr:>9,.0f} | {speedup:>18.2f}x"
        )
