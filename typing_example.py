# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : typing, mypy, 성능 측정 비교 실습
# 기능     : 타입 힌트의 사용에 따른 함수 정의 / timeit을 사용한 성능 비교
# 작성일 : 2026-01-13
# ------------------------------------------------------------

from typing import List

# 1. 두 가지 버전의 함수 정의 (입력 : 정수리스트 -> 출력 : 각 원소의 제곱을 더한 합)
# 1 ) A 버전 : 타입 힌트를 사용하지 않은 함수
def sum_of_squares_a(data):
    total = 0
    for number in data:
        total += number * number
    return total

# 2 ) B 버전 : 타입 힌트를 사용한 함수
def sum_of_squares_b(data: List[int]) -> int:
    total: int = 0
    for number in data:
        total += number * number
    return total

# 2. timeit을 사용하여 두 버전의 실행 성능 비교
if __name__ == "__main__":
    import random
    import timeit

    # 랜덤 정수 리스트 생성
    data = [random.randint(1, 100) for _ in range(1_000_000)]

    # A 버전 성능 측정
    time_a = timeit.timeit(lambda: sum_of_squares_a(data), number=10)
    print(f"A 버전 실행 시간: {time_a:.4f}초")

    # B 버전 성능 측정
    time_b = timeit.timeit(lambda: sum_of_squares_b(data), number=10)
    print(f"B 버전 실행 시간: {time_b:.4f}초")
    
# 3. mypy로 타입 체크 (명령어로 실행)
# -> mypy typing_example.py