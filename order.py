# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : OOP 기반 AI 추천 주문 시스템 설계 실습 (온라인 음료 주문 플랫폼)
# 기능     : 메뉴 정의 / 주문 내역 저장 / 태그 기반 추천 / 총액·평균 계산
# 작성일 : 2026-01-13
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Dict, Optional
from abc import ABC, abstractmethod


# ============================================================
# 1. 도메인 모델(음료)
# ============================================================
@dataclass(frozen=True)
class Beverage:
    name: str
    price: int
    tags: Tuple[str, ...] 

    def __str__(self) -> str:
        return f"{self.name} ({self.price}원) tags={list(self.tags)}"


# ============================================================
# 2. 추천 시스템
# - 최근 주문 1건의 태그와 겹치는 정도로 점수화해 추천
# ============================================================
class Recommender(ABC):
    @abstractmethod
    def recommend(
        self, menu: List[Beverage], orders: List[Beverage], top_k: int = 3
    ) -> List[Beverage]:
        pass


class TagSimilarityRecommender(Recommender):
    def recommend(
        self, menu: List[Beverage], orders: List[Beverage], top_k: int = 3
    ) -> List[Beverage]:
        if not orders:
            return []

        last = orders[-1]
        last_tags = set(last.tags)
        ordered_names = {b.name for b in orders}

        scored: List[Tuple[int, int, Beverage]] = []
        for b in menu:
            if b.name in ordered_names:
                continue

            score = len(last_tags & set(b.tags))
            if score > 0:
                # (유사도 점수, 가격, 음료)로 정렬 기준 구성
                scored.append((score, b.price, b))

        scored.sort(reverse=True)  # score -> price 순으로 내림차순
        return [b for _, _, b in scored[:top_k]]


# ============================================================
# 3. 주문 시스템(클래스)
# - @property로 계산값을 속성처럼 제공
# ============================================================
class OrderSystem:
    def __init__(self, menu: List[Beverage], recommender: Recommender):
        self._menu = menu
        self._orders: List[Beverage] = []
        self._recommender = recommender

        # 검색 최적화: 이름 -> Beverage
        self._menu_by_name: Dict[str, Beverage] = {b.name: b for b in menu}

    # ----------------------------
    # @property: 메뉴/주문 내역 읽기 전용 제공
    # ----------------------------
    @property
    def menu(self) -> List[Beverage]:
        return self._menu

    @property
    def orders(self) -> List[Beverage]:
        return self._orders

    # ----------------------------
    # @property: 계산값(총액/평균/최근 주문)
    # ----------------------------
    @property
    def total_amount(self) -> int:
        return sum(b.price for b in self._orders)

    @property
    def average_amount(self) -> float:
        return self.total_amount / len(self._orders) if self._orders else 0.0

    @property
    def last_order(self) -> Optional[Beverage]:
        return self._orders[-1] if self._orders else None

    # ----------------------------
    # 주문 추가 (도메인 메서드)
    # ----------------------------
    def order(self, beverage_name: str) -> None:
        beverage = self._menu_by_name.get(beverage_name)
        if not beverage:
            raise ValueError(f"메뉴에 없는 음료입니다: {beverage_name}")
        self._orders.append(beverage)

    # ----------------------------
    # 추천 (추천 전략에 위임)
    # ----------------------------
    def get_recommendations(self, top_k: int = 3) -> List[Beverage]:
        return self._recommender.recommend(self._menu, self._orders, top_k=top_k)

    # ----------------------------
    # 출력 유틸
    # ----------------------------
    def print_menu(self) -> None:
        print("\n[메뉴]")
        for i, b in enumerate(self._menu, start=1):
            print(f"{i}. {b}")

    def print_orders(self) -> None:
        print("\n[주문 내역]")
        if not self._orders:
            print("주문 내역이 없습니다.")
            return
        for i, b in enumerate(self._orders, start=1):
            print(f"{i}. {b}")


# ============================================================
# 4. 실행
# ============================================================
if __name__ == "__main__":
    menu = [
        Beverage("아이스 아메리카노", 3000, ("커피", "콜드")),
        Beverage("카페라떼", 3500, ("커피", "밀크")),
        Beverage("녹차", 2800, ("차", "뜨거운")),
        Beverage("허브티", 3000, ("차", "차가운")),
        Beverage("콜드브루", 3800, ("커피", "콜드")),
        Beverage("핫초코", 4000, ("초코", "뜨거운", "밀크")),
    ]

    recommender = TagSimilarityRecommender()
    system = OrderSystem(menu=menu, recommender=recommender)

    # 메뉴 출력
    system.print_menu()

    # ----------------------------
    # 사용자 주문 입력
    # ----------------------------
    print("\n[주문 입력]")
    print("음료 이름을 입력하세요. 종료하려면 'q' 입력")

    while True:
        user_input = input("주문할 음료: ").strip()

        if user_input.lower() == "q":
            break

        try:
            system.order(user_input)
            print(f"주문 완료: {user_input}")
        except ValueError as e:
            print(e)

    # 주문 내역 출력
    system.print_orders()

    # ----------------------------
    # 추천 출력
    # ----------------------------
    print("\n[추천 음료]")
    if system.last_order:
        print(
            f"최근 주문: {system.last_order.name} tags={list(system.last_order.tags)}"
        )

    recs = system.get_recommendations(top_k=3)
    if not recs:
        print("추천할 음료가 없습니다.")
    else:
        for b in recs:
            print(f"- {b.name} ({b.price}원) tags={list(b.tags)}")

    # ----------------------------
    # 결제/통계 (@property 활용)
    # ----------------------------
    print("\n[결제/통계]")
    print(f"총 주문 수량: {len(system.orders)}잔")
    print(f"총 주문 금액: {system.total_amount}원")
    print(f"평균 금액: {system.average_amount:.2f}원")
