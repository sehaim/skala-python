# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : AST 기반 자동 보안 검사기 실습 (Python 1일차 Codelab ①)
# 작성일 : 2025-01-12
# ------------------------------------------------------------

import ast


# ------------------------------------------------------------
# (1) 위험 함수 목록
# - "이 함수가 호출되면 위험하다" 라고 우리가 정해둔 목록
# ------------------------------------------------------------
DANGEROUS_FUNCTIONS = {
    "eval",
    "exec",
}

DANGEROUS_APIS = {
    "os.system",
    "pickle.load",
}


# ------------------------------------------------------------
# (2) 보안 검사기 클래스
# - ast.NodeVisitor를 상속하면 AST를 순회(탐색)할 수 있음
# ------------------------------------------------------------
class SecurityScanner(ast.NodeVisitor):
    def __init__(self, filename: str):
        # 어떤 파일을 검사하고 있는지 저장
        self.filename = filename

        # 탐지 결과를 저장할 리스트
        # 예: ("sample.py", 10, "eval") 같은 형태로 쌓을 것
        self.findings = []

    # --------------------------------------------------------
    # (3) "함수 호출"을 발견할 때마다 자동으로 실행되는 함수
    # - ast.Call 노드를 만날 때 visit_Call이 호출됨
    # --------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        """
        node: 함수 호출 정보를 담고 있는 AST 노드
        """

        # 호출된 함수 이름을 문자열로 만들어 봄
        call_name = self._get_call_name(node.func)

        # 위험 함수인지 확인해서 기록
        if call_name in DANGEROUS_FUNCTIONS or call_name in DANGEROUS_APIS:
            line = node.lineno  # 몇 번째 줄인지
            self.findings.append((self.filename, line, call_name))

        # 아주 중요: 하위 노드도 계속 탐색해야 함
        self.generic_visit(node)

    # --------------------------------------------------------
    # (4) 호출 이름을 구하는 도우미 함수
    # - eval(...) 같은 경우 -> "eval"
    # - os.system(...) 같은 경우 -> "os.system"
    # --------------------------------------------------------
    def _get_call_name(self, func_node):
        # eval(...) 처럼 단순 이름 호출
        if isinstance(func_node, ast.Name):
            return func_node.id

        # os.system(...) 처럼 점(.)이 있는 호출
        if isinstance(func_node, ast.Attribute):
            # func_node.attr = system
            # func_node.value = os (ast.Name)
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"

        # 그 외는 지금 단계에서 처리 안 함
        return ""


# ------------------------------------------------------------
# (5) 파일 하나를 검사하는 함수
# ------------------------------------------------------------
def scan_file(filepath: str):
    # 파일 내용 읽기
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    # 파이썬 코드를 AST로 변환
    tree = ast.parse(source, filename=filepath)

    # 스캐너 생성 후 AST 탐색 시작
    scanner = SecurityScanner(filepath)
    scanner.visit(tree)

    return scanner.findings


# ------------------------------------------------------------
# (6) 결과 출력 함수
# ------------------------------------------------------------
def print_report(findings):
    if not findings:
        print("✅ 위험 함수 사용 없음")
        return

    print("🚨 위험 함수 탐지 결과")
    for filename, line, call_name in findings:
        print(f"- 파일: {filename}, 줄: {line}, 호출: {call_name}")


# ------------------------------------------------------------
# (7) 실행 부분
# - 터미널에서: python scanner.py sample_unsafe.py
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    target_file = sys.argv[1]

    results = scan_file(target_file)
    print_report(results)
