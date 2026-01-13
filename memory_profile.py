# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : 대용량 데이터(1,000만) 처리 시
#           List Comprehension vs Generator Expression
#           메모리 사용량(tracemalloc) 비교 실습
# 작성일 : 2025-01-12
# ------------------------------------------------------------

import tracemalloc
import time


N = 10_000_000


# ============================================================
# 공통: 측정 유틸 (시간 + tracemalloc 메모리)
# ============================================================
def measure(label: str, func):
    """
    func를 실행하면서
    - 처리 시간
    - tracemalloc 기준 현재/최대 메모리 사용량
    을 측정해 출력한다.
    """
    tracemalloc.start()

    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[{label}]")
    print(f" - time   : {elapsed:.3f} s")
    print(f" - current: {current / (1024 * 1024):.2f} MB")
    print(f" - peak   : {peak / (1024 * 1024):.2f} MB")

    return result


# ============================================================
# 1. List Comprehension 방식
# - 1,000만 개 데이터를 리스트로 만들어 메모리에 올린 뒤 처리
# ============================================================
def list_comprehension_pipeline():
    # 1) 데이터 생성(리스트로 모두 생성)  -> 메모리 크게 사용
    data = [i for i in range(N)]

    # 2) 처리 예시: 합계 계산
    # (실습 목적상 간단한 연산으로)
    total = sum(data)

    return total


# ============================================================
# 2. Generator Expression 방식
# - 1,000만 개를 저장하지 않고 필요할 때 하나씩 생성하며 처리
# ============================================================
def generator_expression_pipeline():
    # data는 리스트가 아니라 "생성 규칙"만 갖는 generator
    data = (i for i in range(N))

    # sum이 순회하면서 i를 하나씩 뽑아 쓰고 버림 -> 메모리 적게 사용
    total = sum(data)

    return total


if __name__ == "__main__":
    # 리스트 방식 측정
    total_list = measure("List Comprehension", list_comprehension_pipeline)

    # 제너레이터 방식 측정
    total_gen = measure("Generator Expression", generator_expression_pipeline)
