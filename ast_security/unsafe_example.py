# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : AST 보안 검사기 테스트용 위험 호출 샘플
# 작성일 : 2026-01-12
# ------------------------------------------------------------

import os
import pickle


def trigger_eval():
    # 위험: 문자열을 코드처럼 실행
    eval("print('eval triggered')")


def trigger_exec():
    # 위험: 문자열을 코드처럼 실행
    exec("print('exec triggered')")


def trigger_os_system():
    # 위험: 쉘 명령 실행
    os.system("echo 'os.system triggered'")


def trigger_pickle_load():
    # 위험: 역직렬화 취약점 가능
    with open("dummy.pkl", "rb") as f:
        pickle.load(f)


def main():
    trigger_eval()
    trigger_exec()
    trigger_os_system()
    trigger_pickle_load()


if __name__ == "__main__":
    main()
