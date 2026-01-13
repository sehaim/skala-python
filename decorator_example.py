# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : Python decorator 실습
# 작성일 : 2026-01-13
# ------------------------------------------------------------

import time

# 데코레이터 함수 정의
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# 연산 지연 함수 정의
@measure_time
def slow_function(n):
    import time

    total = 0
    for i in range(n):
        total += i
        time.sleep(0.1);
    return total

# 실행
if __name__ == "__main__":
    slow_function(10)