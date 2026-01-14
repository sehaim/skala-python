# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : Pandas 와 Seaborn을 활용한 데이터 분석 실습
# 기능 : Pandas로 AI 임베딩 전처리를 위한 통계 요약 및 이상치 탐지 / Seaborn으로 변수 간 관계 시각화
# 실습 포인트 :
# 1단계
# - 데이터 전처리
#  . reviews.csv 파일 호출
#  . 결측치 확인/처리
#  . 분포 시각화 및 이상치 탐지
# 2단계
# - 기술 통계 및 시각화
#  . review_length 등 기술 통계 요약
#  . category 별 평균 평점 시각화 (barplot)
#  . 평점과 감성 점수 관계 시각화
#  . 텍스트 길이와 평점의 관계 (boxplot or violinplot)
# 3단계
# - AI 분석을 위한 인사이트 도출
#  . sentiment_score가 높을 수록 평점이 높나?
#  . Review_length가 AI 임베딩 유사도에 영향을 줄 수 있나?
#  . category 별 감성 점수 평균 차이는 존재?
#  . 위의 질문에 대한 그래프 기반 해석 및 3줄 요약 Insight 작성
# 4단계
# - Report 작성
#  . 결과 리포트 작성 시 제목/목차/시각화 포함
#  . 분석 코드와 인사이트를 정리
# 작성일 : 2026-01-14
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

# =========================
# 0. 기본 설정
# =========================

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid")

import koreanize_matplotlib

DATA_PATH = "./data/reviews_1000.csv"

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.head(3))
print("\nColumns:", df.columns.tolist())

# =========================
# 1-1) 결측치 / 중복치 확인
# =========================
print("\n[Missing values]")
print(df.isna().sum().sort_values(ascending=False))

print("\n[Duplicate rows]", df.duplicated().sum())
print("[Duplicate review_id]", df["review_id"].duplicated().sum())

# =========================
# 1-2) 타입 / 값 범위 점검
# =========================
print("\n[Dtypes]")
print(df.dtypes)

# rating 범위 확인 (정상: 1~5)
print("\n[rating value counts]")
print(df["rating"].value_counts().sort_index())

# sentiment_score 분포 범위 확인
print("\n[sentiment_score describe]")
print(df["sentiment_score"].describe())

# =========================
# 1-3) 전처리(결측치 처리 + 파생변수 보강)
#    - 임베딩 유사도 분석 전: review_text는 필수이므로 결측은 제거 권장
#    - sentiment_score는 (1) 제거 or (2) 대체 전략 중 선택
# =========================

df_clean = df.copy()

# (A) review_text 결측 제거 (임베딩 생성 불가)
before = len(df_clean)
df_clean = df_clean.dropna(subset=["review_text"]).reset_index(drop=True)
print(f"\nDropped rows with missing review_text: {before - len(df_clean)}")

# (B) sentiment_score 결측 대체 (카테고리 중앙값으로)
#     - 대체가 싫다면: df_clean = df_clean.dropna(subset=["sentiment_score"])
if df_clean["sentiment_score"].isna().any():
    df_clean["sentiment_score"] = (
        df_clean.groupby("category")["sentiment_score"]
        .transform(lambda s: s.fillna(s.median()))
    )

# (C) review_length가 이미 존재하지만, 안전하게 재계산/검증용 컬럼 추가
df_clean["review_length_calc"] = df_clean["review_text"].astype(str).str.len()

# 길이 불일치 체크(원본 review_length와 계산값이 크게 다르면 확인)
df_clean["length_diff"] = df_clean["review_length_calc"] - df_clean["review_length"]
print("\n[length_diff summary]")
print(df_clean["length_diff"].describe())

# (D) num_words 검증용 재계산 컬럼 추가
df_clean["num_words_calc"] = df_clean["review_text"].astype(str).str.split().str.len()
df_clean["words_diff"] = df_clean["num_words_calc"] - df_clean["num_words"]
print("\n[words_diff summary]")
print(df_clean["words_diff"].describe())

# =========================
# 1-4) 분포 시각화 + 이상치 탐지 준비
# =========================

NUM_COLS = ["review_length", "num_words", "sentiment_score", "rating"]

def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Index:
    """IQR 기반 이상치 인덱스 반환"""
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return s[(s < lower) | (s > upper)].index

def zscore_outliers(s: pd.Series, z: float = 3.0) -> pd.Index:
    """Z-score 기반 이상치 인덱스 반환"""
    s = s.dropna()
    zs = np.abs(stats.zscore(s))
    return s[zs > z].index

# 이상치 요약 테이블
outlier_rows = set()
outlier_report = []

for col in ["review_length", "num_words", "sentiment_score"]:
    idx_iqr = set(iqr_outliers(df_clean[col], k=1.5).tolist())
    idx_z = set(zscore_outliers(df_clean[col], z=3.0).tolist())
    union = idx_iqr.union(idx_z)
    outlier_rows |= union
    outlier_report.append({
        "feature": col,
        "iqr_outliers": len(idx_iqr),
        "zscore_outliers": len(idx_z),
        "union_outliers": len(union),
        "union_ratio": len(union) / len(df_clean)
    })

outlier_report_df = pd.DataFrame(outlier_report).sort_values("union_outliers", ascending=False)
print("\n[Outlier report]")
print(outlier_report_df)

print("\nTotal unique outlier rows (union across features):", len(outlier_rows))

# 필요 시: 이상치 행만 보기
df_outliers = df_clean.loc[sorted(list(outlier_rows))].copy()
print("\n[df_outliers sample]")
print(df_outliers.head(5))

# =========================
# 1-5) 분포 시각화
# =========================
for col in ["review_length", "num_words", "sentiment_score"]:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_clean[col], kde=True)
    plt.title(f"Distribution: {col}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 2.8))
    sns.boxplot(x=df_clean[col])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()

# =========================
# 2) 기술 통계 요약
# =========================
desc = df_clean[NUM_COLS].describe().T
desc["missing"] = df_clean[NUM_COLS].isna().sum()
desc["missing_ratio"] = desc["missing"] / len(df_clean)
print("\n[Describe numeric columns]")
print(desc)

# category별 요약(평점, 감성, 길이)
cat_summary = (
    df_clean.groupby("category")
    .agg(
        n=("review_id", "count"),
        rating_mean=("rating", "mean"),
        rating_median=("rating", "median"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_median=("sentiment_score", "median"),
        len_mean=("review_length", "mean"),
        words_mean=("num_words", "mean"),
    )
    .sort_values("n", ascending=False)
)
print("\n[Category summary]")
print(cat_summary)

# =========================
# 2-1) category 별 평균 평점 시각화 (barplot)
# =========================
plt.figure(figsize=(8, 4))
order = cat_summary.sort_values("rating_mean", ascending=False).index
sns.barplot(data=df_clean, x="category", y="rating", order=order, estimator=np.mean, errorbar="ci")
plt.title("카테고리 별 평균 평점")
plt.xlabel("카테고리")
plt.ylabel("평균 평점")
plt.tight_layout()
plt.show()

# =========================
# 2-2) 평점과 감성 점수 관계 시각화
#    - scatter + regression line
# =========================
plt.figure(figsize=(7, 4))
sns.violinplot(data=df_clean, x="rating", y="sentiment_score", inner="quartile")
plt.title("평점과 감성 점수 관계")
plt.xlabel("감성 점수(sentiment_score)")
plt.ylabel("평점(rating)")
plt.tight_layout()
plt.show()

# =========================
# 2-3) 텍스트 길이와 평점 관계 (boxplot or violinplot)
# =========================
plt.figure(figsize=(7, 4))
sns.boxplot(data=df_clean, x="rating", y="review_length")
plt.title("텍스트 길이와 평점 관계(boxplot)")
plt.xlabel("평점(rating)")
plt.ylabel("텍스트 길이(review_length)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.violinplot(data=df_clean, x="rating", y="review_length", inner="quartile")
plt.title("텍스트 길이와 평점 관계(violinplot)")
plt.xlabel("평점(rating)")
plt.ylabel("텍스트 길이(review_length)")
plt.tight_layout()
plt.show()

# =========================
# 3) AI 분석을 위한 인사이트용 정량 체크(리포트 문장 작성은 X)
#    - Q1 sentiment_score가 높을 수록 rating이 높나? (상관/회귀, 그룹 평균)
#    - Q2 review_length가 임베딩 유사도에 영향을? (길이 proxy로 분산/패턴 확인)
#    - Q3 category 별 sentiment 평균 차이? (ANOVA)
# =========================

# Q1) 상관계수
corr = df_clean[["sentiment_score", "rating", "review_length", "num_words"]].corr(numeric_only=True)
print("\n[Correlation matrix]")
print(corr)

# Q1) sentiment 구간별 평균 rating (binning)
df_clean["sent_bin"] = pd.qcut(df_clean["sentiment_score"], q=5, duplicates="drop")
sent_bin_summary = df_clean.groupby("sent_bin").agg(
    n=("review_id", "count"),
    rating_mean=("rating", "mean"),
    sentiment_mean=("sentiment_score", "mean"),
)
print("\n[Sentiment bin summary]")
print(sent_bin_summary)

plt.figure(figsize=(9, 4))
sns.barplot(data=sent_bin_summary.reset_index(), x="sent_bin", y="rating_mean")
plt.title("Avg Rating by Sentiment Quantile Bin")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Q2) 길이 구간별 rating/sentiment 요약 (임베딩 품질/유사도 영향 가설의 근거로 활용)
df_clean["len_bin"] = pd.qcut(df_clean["review_length"], q=5, duplicates="drop")
len_bin_summary = df_clean.groupby("len_bin").agg(
    n=("review_id", "count"),
    rating_mean=("rating", "mean"),
    sentiment_mean=("sentiment_score", "mean"),
    len_mean=("review_length", "mean"),
)
print("\n[Length bin summary]")
print(len_bin_summary)

plt.figure(figsize=(9, 4))
sns.lineplot(data=len_bin_summary.reset_index(), x="len_bin", y="rating_mean", marker="o")
plt.title("Avg Rating by Review Length Quantile Bin")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Q3) category별 sentiment 평균 차이: ANOVA (간단 검정)
#     - 전제(정규성/등분산)가 완벽하진 않을 수 있어 EDA 수준 참고용
groups = [g["sentiment_score"].dropna().values for _, g in df_clean.groupby("category")]
anova = stats.f_oneway(*groups) if len(groups) >= 2 else None
print("\n[ANOVA: sentiment_score by category]")
print(anova)

plt.figure(figsize=(8, 4))
order2 = cat_summary.sort_values("sentiment_mean", ascending=False).index
sns.barplot(data=df_clean, x="category", y="sentiment_score", order=order2, estimator=np.mean, errorbar="ci")
plt.title("Average Sentiment Score by Category")
plt.xlabel("category")
plt.ylabel("avg sentiment_score")
plt.tight_layout()
plt.show()

# =========================
# 4. 전처리 결과 저장
# =========================
OUT_CLEAN_PATH = "./data/reviews_cleaned.csv"
df_clean.to_csv(OUT_CLEAN_PATH, index=False)
