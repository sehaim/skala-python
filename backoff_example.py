# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : bcrypt 해시 기반의 Rate Limiting 인증 시스템 실습
# 기능 :
#   - bcrypt로 비밀번호 검증
#   - 메모리(Dictionary)에 IP별 실패 횟수 및 차단(until) 시간 기록
#   - 연속 실패 시 Exponential Backoff(지수 백오프)로 재시도 지연을 증가시켜 brute-force를 물리적으로 차단
# 작성일 : 2026-01-14
# ------------------------------------------------------------

import time
from dataclasses import dataclass
from typing import Dict, Optional

import bcrypt


# =========================
# 1. 설정값
# =========================
BASE_DELAY_SEC = 1.0  # 첫 실패 후 지연(초)
MAX_DELAY_SEC = 30.0  # 최대 지연(초)
RESET_ON_SUCCESS = True  # 성공 시 실패횟수 초기화


# =========================
# 2. IP 상태 저장 구조
# =========================
@dataclass
class IpState:
    fail_count: int = 0
    blocked_until: float = 0.0


# 메모리(Dictionary)에 IP별 상태를 기록
ip_state: Dict[str, IpState] = {}


# =========================
# 3. 지수 백오프 계산
# =========================
def compute_backoff_delay(fail_count: int) -> float:
    """
    실패 횟수에 따른 지수 백오프 지연시간을 계산한다.
    - fail_count=1 -> BASE_DELAY
    - fail_count=2 -> BASE_DELAY*2
    - fail_count=3 -> BASE_DELAY*4 ...
    - MAX_DELAY로 상한 적용
    """
    delay = BASE_DELAY_SEC * (2 ** (fail_count - 1))
    return min(delay, MAX_DELAY_SEC)


def remaining_block_seconds(now: float, blocked_until: float) -> float:
    """차단 해제까지 남은 시간(초)"""
    return max(0.0, blocked_until - now)


# =========================
# 4. bcrypt 비밀번호 세팅/검증
# =========================
def hash_password(plain_password: str, rounds: int = 12) -> bytes:
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt)


def verify_password(plain_password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash)


# =========================
# 5. 로그인 방어 로직
# =========================
def login_attempt(ip: str, password_input: str, password_hash: bytes) -> bool:
    now = time.time()

    # IP 상태 가져오기(없으면 생성)
    state = ip_state.get(ip)
    if state is None:
        state = IpState()
        ip_state[ip] = state

    # 1) 현재 차단 상태인지 확인
    remain = remaining_block_seconds(now, state.blocked_until)
    if remain > 0:
        print(
            f"[BLOCK] ip={ip}  남은 대기={remain:.2f}s  (fail_count={state.fail_count})"
        )
        return False

    # 2) bcrypt 검증
    ok = verify_password(password_input, password_hash)

    if ok:
        print(f"[OK]    ip={ip}  로그인 성공 ✅")
        if RESET_ON_SUCCESS:
            state.fail_count = 0
            state.blocked_until = 0.0
            print(f"        → 성공으로 인해 실패 횟수 초기화")
        return True

    # 3) 실패 처리: 실패 횟수 증가 + 지수 백오프 적용
    state.fail_count += 1
    delay = compute_backoff_delay(state.fail_count)
    state.blocked_until = now + delay

    print(
        f"[FAIL]  ip={ip}  로그인 실패 ❌  "
        f"fail_count={state.fail_count}  "
        f"backoff={delay:.2f}s  "
        f"(until={time.strftime('%H:%M:%S', time.localtime(state.blocked_until))})"
    )
    return False


# =========================
# 6. 실습 시나리오
# =========================
def demo_scenario():
    """
    - 정상 사용자 비밀번호는 'correct_password'
    - 공격자는 여러 번 틀린 비밀번호로 시도 -> 지수 백오프가 점점 늘어남
    - 다른 IP는 별도로 카운트됨(분리)
    """
    print("\n" + "=" * 50)
    print("IP 별 로그인 시도 기록")
    print("=" * 50)

    # 실습용 계정 비밀번호 해시 생성
    correct_password = "correct_password"
    password_hash = hash_password(correct_password, rounds=12)

    attacker_ip = "203.0.113.10"
    normal_ip = "198.51.100.7"

    # 공격자: 연속 실패로 백오프 증가 확인
    print("---- 공격자 연속 실패 시도 (지수 백오프 증가) ----")
    for i in range(6):
        login_attempt(
            attacker_ip, password_input="wrong_password", password_hash=password_hash
        )
        time.sleep(0.4)  # 일부러 짧게 재시도 -> 차단 로그가 보이도록

    # 정상 사용자: 다른 IP는 영향을 받지 않음
    print("\n---- 정상 사용자 시도 (다른 IP는 별도 카운트) ----")
    login_attempt(
        normal_ip, password_input="correct_password", password_hash=password_hash
    )

    # 공격자: 잠깐 기다렸다가 재시도(여전히 차단이면 BLOCK 출력)
    print("\n---- 공격자: 잠깐 대기 후 재시도 ----")
    time.sleep(2.0)
    login_attempt(
        attacker_ip, password_input="wrong_password", password_hash=password_hash
    )

    # 공격자가 맞는 비밀번호를 넣어도, 차단 시간 내면 BLOCK (물리적 지연)
    print("\n---- 공격자: 맞는 비밀번호라도 차단 시간 내면 BLOCK ----")
    login_attempt(
        attacker_ip, password_input="correct_password", password_hash=password_hash
    )

    # 차단이 풀릴 때까지 기다렸다가 성공
    print("\n---- 공격자: 차단 해제 후 맞는 비밀번호로 성공 ----")
    # 남은 차단시간 확인
    state = ip_state[attacker_ip]
    remain = remaining_block_seconds(time.time(), state.blocked_until)
    if remain > 0:
        print(f"(대기) 차단 해제까지 {remain:.2f}s 기다립니다...")
        time.sleep(remain + 0.1)

    login_attempt(
        attacker_ip, password_input="correct_password", password_hash=password_hash
    )

    print("\n---- 최종 IP 상태 ----")
    for ip, st in ip_state.items():
        remain2 = remaining_block_seconds(time.time(), st.blocked_until)
        print(f"ip={ip} fail_count={st.fail_count}, blocked_remaining={remain2:.2f}s")


if __name__ == "__main__":
    demo_scenario()
