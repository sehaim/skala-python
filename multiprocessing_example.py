# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : multiprocessing을 활용한 대용량 데이터 처리 실습
# 기능     : 랜덤 난수 생성 / 소수 판별 / 단일 프로세스 vs 멀티프로세스 성능 비교
# 작성일 : 2026-01-13
# ------------------------------------------------------------

from typing import List
import multiprocessing

# 1. random을 사용하여 1,000만 개의 1~100,000 사이 정수 리스트 생성
def generate_large_data(size: int = 10_000_000, lower: int = 1, upper: int = 100_000):
    import random
    return [random.randint(lower, upper) for _ in range(size)]

# 2. 숫자가 소수인지 판별하는 함수
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 3. 두 가지 방식으로 소수의 개수 세기
# 1 ) 단일 프로세스 방식
def count_primes_single_process(data):
    return sum(1 for number in data if is_prime(number))

# 2 ) multiprocessing.Pool 사용
def count_primes_multi_process(data):
    with multiprocessing.Pool(processes=None) as pool:
        results = pool.map(is_prime, data)
    return sum(results)

# 코드 실행
if __name__ == "__main__":
    import time

    # 데이터 생성
    data = generate_large_data()

    # 단일 프로세스 방식
    start_time = time.time()
    prime_count_single = count_primes_single_process(data)
    end_time = time.time()
    print(f"단일 프로세스 소수 개수: {prime_count_single}, 소요 시간: {end_time - start_time:.2f}초")

    # 멀티프로세스 방식
    start_time = time.time()
    prime_count_multi = count_primes_multi_process(data)
    end_time = time.time()
    print(f"멀티프로세스 소수 개수: {prime_count_multi}, 소요 시간: {end_time - start_time:.2f}초")