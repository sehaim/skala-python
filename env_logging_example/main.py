
# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : python-dotenv + logging 환경 변수 기반 로그 실습
# 작성일 : 2025-01-12
# ------------------------------------------------------------

import os
import logging
from dotenv import load_dotenv


# ============================================================
# 1. .env 파일 로드
# ============================================================
load_dotenv()

log_level_str = os.getenv("LOG_LEVEL", "INFO")
app_name = os.getenv("APP_NAME", "MyApp")

# 문자열 로그 레벨을 logging 상수로 변환
log_level = getattr(logging, log_level_str.upper(), logging.INFO)


# ============================================================
# 2. logging 설정
# ============================================================
logger = logging.getLogger(app_name)
logger.setLevel(log_level)

# 로그 포맷: 시간 | 로그레벨 | 파일명:줄번호 | 메시지
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)

# 파일 핸들러
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)

# 핸들러 등록
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# 3. 로그 출력 테스트
# ============================================================
logger.info("앱 실행 시작")
logger.debug("환경 변수 로딩 완료")

try:
    1 / 0
except ZeroDivisionError:
    logger.error("예외 발생 예시", exc_info=True)
