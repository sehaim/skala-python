# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : asyncio 기반 고성능 API Aggregator 구축 실습
# 기능 :
#   - 여러 마이크로서비스(API)에서 데이터를 동시에 가져오는 Aggregator 구현
#   - asyncio.gather로 Concurrent(동시) 호출 구현
#   - Sequential(순차) 방식 대비 Concurrent 방식의 응답 시간(Latency) 이득 측정
# 작성일 : 2026-01-14
# ------------------------------------------------------------


import asyncio
import time
import random


# ============================================================
# 1. Mock Microservices (외부 API 호출 시뮬레이션)
# ============================================================


async def user_service():
    await asyncio.sleep(0.35)
    return "user"


async def orders_service():
    await asyncio.sleep(0.50)
    return "orders"


async def recommendations_service():
    await asyncio.sleep(0.45)
    return "recommendations"


# ============================================================
# 2. 동기(Sequential) 방식 (순차 호출)
# ============================================================


async def aggregate_sequential():
    start = time.perf_counter()

    user = await user_service()
    orders = await orders_service()
    recs = await recommendations_service()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, (user, orders, recs)


# ============================================================
# 3. 비동기(Concurrent) 방식 (asyncio.gather)
# ============================================================


async def aggregate_concurrent():
    start = time.perf_counter()

    results = await asyncio.gather(
        user_service(),
        orders_service(),
        recommendations_service(),
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, results


# ============================================================
# 4. 서비스 호출 실행
# ============================================================


async def main(runs: int = 5):
    print("\n" + "=" * 55)
    print("asyncio 기반 API Aggregator 성능 비교")
    print("=" * 55)

    seq_times = []
    conc_times = []

    for i in range(runs):
        seq_ms, _ = await aggregate_sequential()
        conc_ms, _ = await aggregate_concurrent()

        seq_times.append(seq_ms)
        conc_times.append(conc_ms)

        print(
            f"[Run {i+1}] "
            f"Sequential: {seq_ms:.2f} ms | "
            f"Concurrent: {conc_ms:.2f} ms"
        )

    seq_avg = sum(seq_times) / runs
    conc_avg = sum(conc_times) / runs
    speedup = seq_avg / conc_avg

    print("\n" + "-" * 24 + "Summary" + "-" * 24)
    print(f"동기(Sequential) 방식 평균 실행 시간 : {seq_avg:.2f} ms")
    print(f"비동기(Concurrent) 방식 평균 실행 시간 : {conc_avg:.2f} ms")
    print(f"Speedup        : {speedup:.2f}x")
    print("-" * 56 + "\n")


if __name__ == "__main__":
    asyncio.run(main(runs=5))
