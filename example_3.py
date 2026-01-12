# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : Python 제너레이터 기반 메모리 절약형 로직 응용 실습
# 작성일 : 2025-01-12
# ------------------------------------------------------------

import sys
import time


N = 1_000_000


# ============================================================
# 1. 제너레이터 함수: 0 이상 n 미만의 짝수만 제곱
# ============================================================
def even_square_gen(n: int):
    for x in range(0, n, 2):
        yield x * x


# ============================================================
# 2. 제너레이터를 이용해 총합 계산
# ============================================================
def sum_even_squares_with_generator(n: int) -> int:
    return sum(even_square_gen(n))


# 리스트 방식
def sum_even_squares_with_list(n: int) -> int:
    squares = [x * x for x in range(0, n, 2)]
    return sum(squares)


# ============================================================
# 3. 메모리 사용량, 처리 속도 비교 (time 모듈)
# ============================================================
if __name__ == "__main__":
    # -----------------------------
    # 메모리 사용량 비교
    # -----------------------------
    gen_obj = even_square_gen(N)
    gen_size = sys.getsizeof(gen_obj)

    squares_list = [x * x for x in range(0, N, 2)]
    list_size = sys.getsizeof(squares_list)

    print("[메모리 사용량]")
    print(f" - 제너레이터 객체 크기: {gen_size:,} bytes")
    print(f" - 리스트 객체 크기     : {list_size:,} bytes")

    # -----------------------------
    # 처리 속도 비교
    # -----------------------------
    start = time.perf_counter()
    total_gen = sum_even_squares_with_generator(N)
    gen_time = time.perf_counter() - start

    start = time.perf_counter()
    total_list = sum_even_squares_with_list(N)
    list_time = time.perf_counter() - start

    print("[처리 속도]")
    print(f" - 제너레이터 처리 속도: {gen_time:.6f} s")
    print(f" - 리스트 처리 속도     : {list_time:.6f} s")
