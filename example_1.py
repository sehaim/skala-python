# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : Python 리스트 / 딕셔너리 실습
# 작성일 : 2026-01-12
# ------------------------------------------------------------

from collections import defaultdict
from typing import List, Dict, Tuple


employees: List[Dict[str, object]] = [
    {"name": "Alice", "department": "Engineering", "age": 30, "salary": 85000},
    {"name": "Bob", "department": "Marketing", "age": 25, "salary": 60000},
    {"name": "Charlie", "department": "Engineering", "age": 35, "salary": 95000},
    {"name": "David", "department": "HR", "age": 45, "salary": 70000},
    {"name": "Eve", "department": "Engineering", "age": 28, "salary": 78000},
]


# 1. Engineering 부서이면서 salary >= 80000인 직원 이름 출력
def high_salary_engineers(data: List[Dict[str, object]]) -> List[str]:
    return [
        e["name"]
        for e in data
        if e["department"] == "Engineering" and e["salary"] >= 80_000
    ]


# 2. 30세 이상 직원의 (이름, 부서) 튜플 리스트 출력
def employees_over_30(data: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    return [(e["name"], e["department"]) for e in data if e["age"] >= 30]


# 3. 급여 기준 내림차순 정렬 후 상위 3명 출력
def top_3_by_salary(data: List[Dict[str, object]]) -> List[Tuple[str, int]]:
    return sorted(
        ((e["name"], e["salary"]) for e in data), key=lambda x: x[1], reverse=True
    )[:3]

# 4. 부서별 평균 급여 출력
def average_salary_by_department(
    data: List[Dict[str, object]]
) -> Dict[str, float]:

    # {부서명: [총급여, 인원수]}
    summary = defaultdict(lambda: [0, 0])

    for e in data:
        summary[e["department"]][0] += e["salary"]
        summary[e["department"]][1] += 1

    return {
        dept: total / count
        for dept, (total, count) in summary.items()
    }


if __name__ == "__main__":
    print("1. ", high_salary_engineers(employees))
    print("2. ", employees_over_30(employees))
    print("3. ", top_3_by_salary(employees))
    print("4. ")
    for dept, avg in average_salary_by_department(employees).items():
        print(f"{dept}: {avg:}")