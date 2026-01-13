# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : AST 기반 자동 보안 검사기 실습 
# 작성일 : 2025-01-12
# ------------------------------------------------------------

import ast
import os
import logging
from dotenv import load_dotenv


# ============================================================
# .env 로드 + logging 설정
# ============================================================
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)

logger = logging.getLogger("security_scanner")
logger.setLevel(LOG_LEVEL)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

for handler in (
    logging.StreamHandler(),
    logging.FileHandler("scanner.log", encoding="utf-8"),
):
    handler.setLevel(LOG_LEVEL)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 중복 로그 방지
logger.propagate = False


# ============================================================
# 위험 함수 목록
# ============================================================
DANGEROUS = {"eval", "exec", "os.system", "pickle.load"}


# ============================================================
# AST 보안 검사기
# ============================================================
class SecurityScanner(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.findings = []

    def visit_Call(self, node: ast.Call):
        name = self._call_name(node.func)
        if name in DANGEROUS:
            self.findings.append((self.filename, node.lineno, name))
        self.generic_visit(node)

    def _call_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return ""


# ============================================================
# 파일 스캔 + 로그 출력
# ============================================================
def scan_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    scanner = SecurityScanner(path)
    scanner.visit(tree)
    return scanner.findings


def log_report(findings):
    if not findings:
        logger.info("✅ 위험 함수 사용 없음")
        return

    logger.warning("🚨 위험 함수 탐지 결과")
    for file, line, call in findings:
        logger.warning(f"{file}:{line} | {call}")


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    import sys

    results = scan_file(sys.argv[1])
    log_report(results)
