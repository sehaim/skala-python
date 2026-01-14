# 고객 리뷰 데이터 EDA 리포트

**Pandas & Seaborn 기반 AI 임베딩 전처리 분석**

- 작성자: 임세하
- 작성일: 2026-01-14
- 목적: 고객 리뷰 데이터를 AI 임베딩 및 유사도 분석에 활용하기 전, 데이터의 분포와 패턴을 이해하고 분석 인사이트를 도출합니다.

---

## 목차

1. [데이터 개요](#1-데이터-개요)
2. [데이터 전처리](#2-데이터-전처리)  
    2.1 [데이터 로드](#21-데이터-로드)  
    2.2 [결측치 확인 및 처리](#22-결측치-확인-및-처리)  
    2.3 [분포 시각화 및 이상치 탐지](#23-분포-시각화-및-이상치-탐지)
3. [기술 통계 및 시각화](#3-기술-통계-및-시각화)  
    3.1 [주요 변수 기술 통계 요약](#31-주요-변수-기술-통계-요약)  
    3.2 [카테고리별 평균 평점 분석](#32-카테고리별-평균-평점-분석)  
    3.3 [평점과 감성 점수 관계 분석](#33-평점과-감성-점수-관계-분석)  
    3.4 [텍스트 길이와 평점 관계 분석](#34-텍스트-길이와-평점-관계-분석)
4. [AI 분석을 위한 인사이트 도출](#4-ai-분석을-위한-인사이트-도출)  
    4.1 [감성 점수와 평점의 관계](#41-감성-점수와-평점의-관계)  
    4.2 [리뷰 길이와 임베딩 유사도 영향 가능성](#42-리뷰-길이와-임베딩-유사도-영향-가능성)  
    4.3 [카테고리별 감성 점수 차이 분석](#43-카테고리별-감성-점수-차이-분석)
5. [핵심 인사이트 요약](#5-핵심-인사이트-요약)
6. [결론](#6-결론)

---

## 1. 데이터 개요

본 분석은 제공된 `reviews_1000.csv` 데이터를 기반으로 수행되었습니다. 고객 리뷰를 AI 임베딩 기반 추천 시스템에 활용하기 전 단계로서 EDA(Exploratory Data Analysis)를 진행했습니다.

- 전체 데이터 수: **1,000건**
- 주요 컬럼:
  - `review_id` : 리뷰 고유 ID
  - `product_id` : 상품 ID
  - `category`: 상품 카테고리
  - `review_text`: 리뷰 텍스트
  - `review_length`: 리뷰 길이
  - `num_words`: 단어 수
  - `sentiment_score`: 감성 점수 (-1 ~ 1)
  - `rating`: 평점 (1 ~ 5)

---

## 2. 데이터 전처리

### 2.1 데이터 로드

- 코드
  ```python
  df = pd.read_csv(DATA_PATH)
  print("Shape:", df.shape)
  print(df.head(3))
  print("\nColumns:", df.columns.tolist())
  ```
- 결과

  ![데이터 로드 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/data_load.png)

- 데이터 구조 및 컬럼 타입을 확인하여 분석 가능 여부를 점검했고, 이상 없음을 확인했습니다.

### 2.2 결측치 확인 및 처리

- 코드

  ```python
    # 결측치 확인
    print(df.isna().sum().sort_values(ascending=False))

    df_clean = df.copy()

    # review_text 결측 제거
    df_clean = df_clean.dropna(subset=["review_text"]).reset_index(drop=True)

    # sentiment_score 결측 대체
    df_clean["sentiment_score"] = (df_clean.groupby("category")["sentiment_score"].transform(lambda s: s.fillna(s.median())))
  ```

- 결과

  ![데이터 결측치 확인 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/missing_values.png)

- 결측치 처리 전략은 다음과 같습니다.

  - `review_text` : AI 임베딩 생성이 불가능하므로 해당 행 제거
  - `sentiment_score` : 동일 카테고리 내 중앙값으로 대체

- 결측치 처리 후 분석 대상 데이터는 **995건**입니다.

### 2.3 분포 시각화 및 이상치 탐지

AI 임베딩 및 유사도 분석 전,
리뷰 데이터의 수치적 특성과 이상치 존재 여부를 확인하기 위해
분포 시각화 및 이상치 탐지를 수행했습니다.

- #### 파생 변수 생성 및 데이터 검증

  기존에 제공된 텍스트 길이 관련 컬럼의 신뢰성을 검증하기 위해, 리뷰 텍스트(`review_text`)를 기반으로 파생 변수를 생성했습니다.

  - 코드

    ```python
    f_clean["review_length_calc"] = df_clean["review_text"].astype(str).str.len()

    # 길이 불일치 체크
    df_clean["length_diff"] = df_clean["review_length_calc"] - df_clean["review_length"]
    print("\n[length_diff summary]")
    print(df_clean["length_diff"].describe())

    # num_words 검증용 재계산 컬럼 추가
    df_clean["num_words_calc"] = df_clean["review_text"].astype(str).str.split().str.len()
    df_clean["words_diff"] = df_clean["num_words_calc"] - df_clean["num_words"]
    print("\n[words_diff summary]")
    print(df_clean["words_diff"].describe())
    ```

  - 결과

    ![데이터 검증 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/data_check.png)

  - 기존 컬럼과 재계산된 값 간의 차이를 비교한 결과, 데이터 전반에서 큰 불일치는 관찰되지 않았으며, 제공된 `review_length`, `num_words` 컬럼은 신뢰 가능한 것으로 판단했습니다.

- #### 분포 시각화

  주요 수치형 변수의 분포를 확인하기 위해 히스토그램과 박스플롯을 활용했습니다.

  - 코드

    ```python
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
    ```

  - 결과

    - **review_length**

      ![review_length 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length_histogram.png)

      ![review_length boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length_boxplot.png)

      - 리뷰 길이는 약 40~200자 범위 내에서 분포하며, 비교적 넓은 분산을 보였습니다.
      - 히스토그램 기준 중간 길이(약 100~150자)에 리뷰가 많이 분포하는 경향이 나타났습니다.
      - 박스플롯 상 극단적으로 튀는 이상치는 확인되지 않았으며, 전체적으로 안정적인 분포를 유지하고 있습니다.

    - **rating**

      ![rating 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating_histogram.png)
      ![rating boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating_boxplot.png)

      - 평점은 1~5의 이산값을 가지며, 전반적으로 3~5 구간에 비교적 많이 분포했습니다.
      - 박스플롯 기준 중앙값은 약 3 수준으로 확인되며, 특정 평점에 과도하게 치우친 분포는 관찰되지 않았습니다.
      - 이상치로 판단할 만한 극단적인 값은 존재하지 않았습니다.

    - **sentiment_score**

      ![sentiment_score 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/sentiment_score_histogram.png)

      ![sentiment_score boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/sentiment_score_boxplot.png)

      - 감성 점수는 -1 ~ 1 범위 내에서 분포하며, 음수와 양수 영역 모두 고르게 나타났습니다.
      - 히스토그램에서는 완만한 이중 봉우리 형태가 관찰되어, 긍·부정 리뷰가 모두 일정 비율 존재함을 확인할 수 있습니다.
      - 박스플롯 기준에서도 이상치로 판단될 만한 값은 확인되지 않았습니다.

- #### 이상치 탐지 (IQR / Z-score)

  분포 시각화 이후, IQR 및 Z-score 기준을 활용하여 이상치를 정량적으로 탐지했습니다.

  - 코드

    ```python
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
    ```

  - 출력 결과

    ![이상치 탐지 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/IQR_ZScore.png)

  - IQR 및 Z-score 기준 모두에서 이상치로 판단되는 데이터는 발견되지 않았고, 데이터 전반이 정상 범위 내에 분포하고 있음을 확인했습니다. 따라서 이상치 제거 없이 전체 데이터를 분석 대상으로 유지했습니다.

---

## 3. 기술 통계 및 시각화

### 3.1 주요 변수 기술 통계 요약

리뷰 데이터의 전반적인 수치적 특성을 파악하기 위해 주요 변수의 기술 통계를 산출했습니다.

- 코드

  ```python
  NUM_COLS = ["review_length", "num_words", "sentiment_score", "rating"]

  desc = df_clean[NUM_COLS].describe().T
  desc["missing"] = df_clean[NUM_COLS].isna().sum()
  desc["missing_ratio"] = desc["missing"] / len(df_clean)

  print("\n[Describe numeric columns]")
  print(desc)
  ```

- 결과
  | 변수 | 평균 | 중앙값 | 표준편차 |
  | --------------- | ------- | ------ | -------- |
  | review_length | 약 124 | 125 | 41.65 |
  | num_words | 약 29 | 28 | 11.31 |
  | sentiment_score | 약 0.02 | 0.03 | 0.61 |
  | rating | 약 3.21 | 3 | 1.35 |

---

### 3.2 카테고리별 평균 평점 분석 (barplot)

카테고리별 평점 평균을 비교하여 상품 유형에 따른 평점 차이를 확인했습니다.

- 코드
    ```python
    plt.figure(figsize=(8, 4))
    order = cat_summary.sort_values("rating_mean", ascending=False).index
    sns.barplot(data=df_clean, x="category", y="rating", order=order, estimator=np.mean, errorbar="ci")
    plt.title("카테고리 별 평균 평점")
    plt.xlabel("카테고리")
    plt.ylabel("평균 평점")
    plt.tight_layout()
    plt.show()
    ```

- 결과
    ![카테고리 별 평균 평점](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/category-avg_rating.png)

- `home`, `fashion` 카테고리가 상대적으로 높은 평균 평점을 보입니다.
- 카테고리 간 평균 평점 차이는 크지 않으나 순위 차이는 존재합니다.

---

### 3.3 평점과 감성 점수 관계 분석 (violinplot)

평점(`rating`)과 감성 점수(`sentiment_score`)의 관계를 시각화하여, 감성 점수가 평점에 영향을 주는지 확인했습니다.

- 코드
    ```python
    plt.figure(figsize=(7, 4))
    sns.violinplot(data=df_clean, x="rating", y="sentiment_score", inner="quartile")
    plt.title("평점과 감성 점수 관계")
    plt.xlabel("감성 점수(sentiment_score)")
    plt.ylabel("평점(rating)")
    plt.tight_layout()
    plt.show()
    ```

- 결과
    ![평점과 감성 점수 관계](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating-sentiment_score.png)

- 감성 점수가 높아질수록 평점이 함께 증가하는 경향을 보입니다.

---

### 3.4 리뷰 길이와 평점 관계 분석 (boxplot)

리뷰 길이(review_length)가 평점과 관계가 있는지 확인하기 위해 분포 기반으로 비교했습니다.

- 코드
    ```python
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_clean, x="rating", y="review_length")
    plt.title("리뷰 길이와 평점 관계(boxplot)")
    plt.xlabel("평점(rating)")
    plt.ylabel("리뷰 길이(review_length)")
    plt.tight_layout()
    plt.show()
    ```

- 결과

    ![리뷰 길이와 평점 관계](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length-avg_rating.png)

- 리뷰 길이와 평점 간 직접적인 관계는 뚜렷하지 않습니다.

---

## 4. AI 분석을 위한 인사이트 도출

### 4.1 감성 점수와 평점의 관계

감성 점수가 높을수록 평점이 높아지는지를 정량적으로 확인하기 위해 상관계수 및 분위수 구간별 평균 평점을 계산했습니다. 또한 감성 점수를 분위수(5등분)로 나누어, 구간별 평균 평점 변화를 확인했습니다.

- 코드

  ```python
  corr = df_clean[["sentiment_score", "rating", "review_length", "num_words"]].corr(numeric_only=True)

  print("\n[Correlation matrix]")
  print(corr)

  df_clean["sent_bin"] = pd.qcut(df_clean["sentiment_score"], q=5, duplicates="drop")
  sent_bin_summary = df_clean.groupby("sent_bin").agg(
  n=("review_id", "count"),
  rating_mean=("rating", "mean"),
  sentiment_mean=("sentiment_score", "mean"),)

  plt.figure(figsize=(9, 4))
  sns.barplot(data=sent_bin_summary.reset_index(), x="sent_bin", y="rating_mean")
  plt.title("감성 점수 구간별 평균 평점")
  plt.xlabel("감성 점수 구간")
  plt.ylabel("평균 평점")
  plt.xticks(rotation=25, ha="right")
  plt.tight_layout()
  plt.show()
  ```

- 결과

  ![속성 간 상관계수 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/correlation_matrix.png)

  - 상관계수 계산 결과, `sentiment_score`와 `rating`은 약 `0.73`의 강한 양의 상관관계를 보였습니다.

  ![감성 점수와 평점의 관계 분석 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/sentiment_score_bin-avg_rating.png)

  - 감성 점수 분위수가 높아질수록 평균 평점이 단계적으로 상승하여, 감성 점수가 평점 및 추천 모델에 중요한 신호로 작용할 가능성이 높습니다.

### 4.2 리뷰 길이와 임베딩 유사도 영향 가능성

리뷰 길이는 임베딩 입력 텍스트의 정보량과 직결될 수 있으므로, 길이 구간별 평점 및 감성 점수 평균을 확인했습니다.

- 코드

  ```python
  df_clean["len_bin"] = pd.qcut(df_clean["review_length"], q=5, duplicates="drop")

  len_bin_summary = df_clean.groupby("len_bin").agg(
  n=("review_id", "count"),
  rating_mean=("rating", "mean"),
  sentiment_mean=("sentiment_score", "mean"),
  len_mean=("review_length", "mean"),)

  plt.figure(figsize=(9, 4))
  sns.barplot(data=len_bin_summary.reset_index(), x="len_bin", y="rating_mean")
  plt.title("리뷰 길이 구간별 평균 평점")
  plt.xticks(rotation=25, ha="right")
  plt.tight_layout()
  plt.show()
  ```

- 결과

  ![리뷰 길이와 임베딩 유사도 영향 가능성 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length_bin-avg_rating.png)

- 리뷰 길이 구간별 평균 평점은 큰 폭으로 단조 증가/감소하지 않아, 길이가 평점에 직접적 영향을 주는 변수로 보이진 않았습니다. 다만 임베딩 관점에서는 너무 짧은 텍스트가 정보 부족을 유발할 수 있고, 너무 긴 텍스트는 노이즈 증가/비용 증가를 야기할 수 있으므로 적정 길이 범위를 고려할 필요가 있습니다.

### 4.3 카테고리별 감성 점수 차이 분석

카테고리별로 감성 점수 평균 차이가 존재하는지 검정하기 위해 일원분산분석(ANOVA)과 평균 비교 시각화를 함께 수행했습니다.

- 코드

  ```python
    groups = [g["sentiment_score"].dropna().values for _, g in df_clean.groupby("category")]
    anova = stats.f_oneway(*groups) if len(groups) >= 2 else None

    print("\n[ANOVA: Sentiment Score by category]")
    print(anova)

    plt.figure(figsize=(8, 4))
    order2 = cat_summary.sort_value("sentiment_mean", ascending=False).index
    sns.barplot(data=df_clean, x="category",        y="sentiment_score", order=order2, estimator=np.mean, errorbar="ci")
    plt.title("카테고리 별 감성 점수 평균")
    plt.xlabel("카테고리")
    plt.ylabel("감성 점수 평균")
    plt.tight_layout()
    plt.show()
  ```

- 결과

  ![카테고리별 감성 점수 차이 anova 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/anova.png)

  - ANOVA 수행 결과, F 통계량은 약 0.247, p-value는 약 0.863으로 나타났습니다. 이는 유의수준 0.05 기준에서 카테고리별 감성 점수 평균 차이가 통계적으로 유의하지 않음을 의미합니다.

  ![카테고리별 감성 점수 차이 분석 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/category-avg_sentiment.png)

  - 카테고리별 평균 감성 점수를 시각화한 결과에서도, 각 카테고리의 평균값은 유사한 수준을 보였으며 신뢰구간이 크게 겹쳐 뚜렷한 차이를 확인하기 어려웠습니다.

- 종합하면, 감성 점수는 특정 카테고리에 의해 크게 구분되기보다는
  리뷰 텍스트 자체의 내용적 특성을 더 강하게 반영하는 변수로 해석할 수 있습니다. 따라서 AI 임베딩 기반 분석 및 추천 시스템에서는 카테고리 정보보다 리뷰 문장의 의미 표현이 더 중요한 역할을 할 가능성이 있습니다.

---

## 5. 핵심 인사이트 요약

본 EDA 분석을 통해 AI 임베딩 기반 추천 시스템 구축에 활용할 수 있는 주요 인사이트는 다음과 같습니다.

1. **감성 점수는 평점과 강한 양의 상관관계를 보인다.**  
   상관계수 분석 결과, `sentiment_score`와 `rating` 간 상관계수는 약 0.73으로 나타났으며,  
   감성 점수 분위수가 높아질수록 평균 평점이 단계적으로 상승하는 경향이 확인되었습다.  
   이는 감성 점수가 사용자 만족도를 잘 반영하는 핵심 변수임을 시사합니다.

2. **리뷰 길이는 평점과 직접적인 관계는 약하지만, 임베딩 품질에는 영향을 줄 수 있다.**  
   리뷰 길이 구간별 평균 평점에는 뚜렷한 단조 증가·감소 패턴이 나타나지 않았으나,  
   너무 짧은 리뷰는 정보 부족을, 너무 긴 리뷰는 노이즈 및 연산 비용 증가를 유발할 수 있습니다.  
   따라서 임베딩 생성 시 적정 리뷰 길이 범위를 고려하는 전략이 필요합니다.

3. **카테고리별 감성 점수 평균 차이는 통계적으로 유의하지 않다.**  
   ANOVA 분석 결과(p-value = 0.863), 카테고리 간 감성 점수 평균 차이는 유의하지 않았으며,  
   시각화 결과에서도 각 카테고리의 평균 및 신뢰구간이 크게 겹쳤습니다.  
   이는 감성 점수가 상품 카테고리보다는 리뷰 텍스트 자체의 내용에 더 크게 의존함을 의미합니다.

---

## 6. 결론

본 분석에서는 고객 리뷰 데이터를 AI 임베딩 및 유사도 분석에 활용하기 위한 사전 단계로서 Pandas와 Seaborn을 활용한 EDA를 수행했습니다.

데이터 전처리 과정에서 결측치 처리 및 파생 변수 검증을 통해 데이터의 신뢰성을 확보했으며, 분포 시각화와 이상치 탐지를 통해 전반적으로 안정적인 데이터 특성을 확인했습니다.

분석 결과, 감성 점수는 평점과 강한 연관성을 가지는 핵심 변수로 확인되었으며,  
카테고리보다는 리뷰 텍스트의 의미적 정보가 감성 및 평점에 더 큰 영향을 미치는 것으로 해석됩니다.  
또한 리뷰 길이는 평점과 직접적인 관계는 약하지만, 임베딩 품질 관점에서 중요한 고려 요소임을 확인했습니다.

이러한 결과를 바탕으로, 향후 AI 임베딩 기반 추천 시스템에서는  
카테고리 정보보다는 리뷰 텍스트의 의미 표현과 감성 정보를 중심으로 한 모델 설계가 효과적일 것으로 판단됩니다.  
본 EDA 결과는 이후 LLM 임베딩 생성, 유사도 계산 및 추천 로직 설계의 기초 자료로 활용될 수 있습니다.
