# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : 대용량 데이터(1,000만) 처리 시
#           List Comprehension vs Generator Expression
#           메모리 사용량(tracemalloc) 비교 실습
# 작성일 : 2026-01-12
# ------------------------------------------------------------

import time
import tracemalloc
from functools import wraps

N = 10_000_000


# =============================================================
# 공통: 메모리 계산 데코레이터
# =============================================================
def profile(label: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()

            result = func(*args, **kwargs)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"[{label}] {peak:,} bytes")
            return result, peak

        return wrapper

    return decorator


# =============================================================
# 1. List Comprehension (Eager Evaluation)
# - 리스트를 먼저 모두 만들어 메모리에 올린 뒤 sum
# =============================================================
@profile("List Comprehension")
def use_list_comprehension(n: int) -> int:
    return sum([i for i in range(n)])


# =============================================================
# 2. Generator Expression (Lazy Evaluation)
# - 값을 미리 저장하지 않고 필요할 때 하나씩 생성하며 sum
# =============================================================
@profile("Generator Expression")
def use_generator_expression(n: int) -> int:
    return sum(i for i in range(n))


# =============================================================
# 3. 실행
# =============================================================
if __name__ == "__main__":

    print("-" * 56)
    print(f"List Comprehension vs Generator Expression 메모리 테스트")
    print("-" * 56)

    list_total, list_peak = use_list_comprehension(N)
    gen_total, gen_peak = use_generator_expression(N)
