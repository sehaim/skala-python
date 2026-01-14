# ------------------------------------------------------------
# 작성자 : 임세하
# 작성목적 : Pandas 와 Seaborn을 활용한 데이터 분석 실습
# 기능 : Pandas로 AI 임베딩 전처리를 위한 통계 요약 및 이상치 탐지 / Seaborn으로 변수 간 관계 시각화
# 실습 포인트 : AI 분석을 위한 인사이트 도출
#           - sentiment_score가 높을 수록 평점이 높은가?
#           - Review_length가 AI 임베딩 유사도에 영향을 줄 수 있는가?
#           - category 별 감성 점수 평균 차이는 존재하는가?
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
# =========================
df_clean = df.copy()

# (A) review_text 결측 제거 (임베딩 생성 불가)
before = len(df_clean)
df_clean = df_clean.dropna(subset=["review_text"]).reset_index(drop=True)
print(f"\nDropped rows with missing review_text: {before - len(df_clean)}")

# (B) sentiment_score 결측 대체 (카테고리 중앙값으로)
if df_clean["sentiment_score"].isna().any():
    df_clean["sentiment_score"] = df_clean.groupby("category")[
        "sentiment_score"
    ].transform(lambda s: s.fillna(s.median()))

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
# 1-4)이상치 탐지
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
    outlier_report.append(
        {
            "feature": col,
            "iqr_outliers": len(idx_iqr),
            "zscore_outliers": len(idx_z),
            "union_outliers": len(union),
            "union_ratio": len(union) / len(df_clean),
        }
    )

outlier_report_df = pd.DataFrame(outlier_report).sort_values(
    "union_outliers", ascending=False
)
print("\n[Outlier report]")
print(outlier_report_df)

print("\nTotal unique outlier rows (union across features):", len(outlier_rows))

# =========================
# 1-5) 분포 시각화
# =========================
for col in ["review_length", "rating", "sentiment_score"]:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_clean[col], kde=True)
    plt.title(f"분포 시각화(히스토그램): {col}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 2.8))
    sns.boxplot(x=df_clean[col])
    plt.title(f"분포 시각화(boxplot): {col}")
    plt.tight_layout()
    plt.show()

# =========================
# 2. 기술 통계 요약
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

# =========================
# 2-1) category 별 평균 평점 시각화
# =========================
plt.figure(figsize=(8, 4))
order = cat_summary.sort_values("rating_mean", ascending=False).index
sns.barplot(
    data=df_clean,
    x="category",
    y="rating",
    order=order,
    estimator=np.mean,
    errorbar="ci",
)
plt.title("카테고리 별 평균 평점")
plt.xlabel("카테고리")
plt.ylabel("평균 평점")
plt.tight_layout()
plt.show()

# =========================
# 2-2) 평점과 감성 점수 관계 시각화
# =========================
plt.figure(figsize=(7, 4))
sns.violinplot(data=df_clean, x="rating", y="sentiment_score", inner="quartile")
plt.title("평점과 감성 점수 관계")
plt.xlabel("감성 점수(sentiment_score)")
plt.ylabel("평점(rating)")
plt.tight_layout()
plt.show()

# =========================
# 2-3) 리뷰 길이와 평점 관계 시각화
# =========================
plt.figure(figsize=(7, 4))
sns.boxplot(data=df_clean, x="rating", y="review_length")
plt.title("리뷰 길이와 평점 관계(boxplot)")
plt.xlabel("평점(rating)")
plt.ylabel("리뷰 길이(review_length)")
plt.tight_layout()
plt.show()

# =========================
# 3) AI 분석을 위한 인사이트용 정량 체크
# =========================
# Q1) 상관계수
corr = df_clean[["sentiment_score", "rating", "review_length", "num_words"]].corr(
    numeric_only=True
)
print("\n[Correlation matrix]")
print(corr)

# Q1) sentiment 구간별 평균 rating (binning)
df_clean["sent_bin"] = pd.qcut(df_clean["sentiment_score"], q=5, duplicates="drop")
sent_bin_summary = df_clean.groupby("sent_bin").agg(
    n=("review_id", "count"),
    rating_mean=("rating", "mean"),
    sentiment_mean=("sentiment_score", "mean"),
)

plt.figure(figsize=(9, 4))
sns.barplot(data=sent_bin_summary.reset_index(), x="sent_bin", y="rating_mean")
plt.title("감성 점수 구간별 평균 평점")
plt.xlabel("감성 점수 구간")
plt.ylabel("평균 평점")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Q2) 리뷰 길이 구간별 rating/sentiment 요약
df_clean["len_bin"] = pd.qcut(df_clean["review_length"], q=5, duplicates="drop")
len_bin_summary = df_clean.groupby("len_bin").agg(
    n=("review_id", "count"),
    rating_mean=("rating", "mean"),
    sentiment_mean=("sentiment_score", "mean"),
    len_mean=("review_length", "mean"),
)

plt.figure(figsize=(9, 4))
sns.barplot(data=len_bin_summary.reset_index(), x="len_bin", y="rating_mean")
plt.title("리뷰 길이 구간별 평균 평점")
plt.xlabel("리뷰 길이 구간")
plt.ylabel("평균 평점")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Q3) category별 sentiment 평균 차이
groups = [g["sentiment_score"].dropna().values for _, g in df_clean.groupby("category")]
anova = stats.f_oneway(*groups) if len(groups) >= 2 else None
print("\n[ANOVA: Sentiment Score by Category]")
print(anova)

plt.figure(figsize=(8, 4))
order2 = cat_summary.sort_values("sentiment_mean", ascending=False).index
sns.barplot(
    data=df_clean,
    x="category",
    y="sentiment_score",
    order=order2,
    estimator=np.mean,
    errorbar="ci",
)
plt.title("카테고리 별 감성 점수 평균")
plt.xlabel("카테고리")
plt.ylabel("감성 점수 평균")
plt.tight_layout()
plt.show()

# =========================
# 4. 전처리 결과 저장
# =========================
OUT_CLEAN_PATH = "./data/reviews_cleaned.csv"
df_clean.to_csv(OUT_CLEAN_PATH, index=False)
