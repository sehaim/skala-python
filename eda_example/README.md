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

본 분석은 제공된 `reviews_1000.csv` 데이터를 기반으로 수행되었습니다. 고객 리뷰를 AI 임베딩 기반 추천 시스템에 활용하기 전 단계로서 EDA(Exploratory Data Analysis)를 진행하였습니다.

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
- 출력 결과

  ![데이터 로드 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/data_load.png)

- 데이터 구조 및 컬럼 타입을 확인하여 분석 가능 여부를 점검하였고, 이상 없음을 확인했습니다.

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

- 출력 결과

  ![데이터 결측치 확인 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/missing_values.png)

- 결측치 처리 전략은 다음과 같습니다.
    - `review_text` : AI 임베딩 생성이 불가능하므로 해당 행 제거
    - `sentiment_score` : 동일 카테고리 내 중앙값으로 대체

- 결측치 처리 후 분석 대상 데이터는 **995건**입니다.

### 2.3 분포 시각화 및 이상치 탐지

AI 임베딩 및 유사도 분석 전,
리뷰 데이터의 수치적 특성과 이상치 존재 여부를 확인하기 위해
분포 시각화 및 이상치 탐지를 수행하였습니다.

- #### 파생 변수 생성 및 데이터 검증

    기존에 제공된 텍스트 길이 관련 컬럼의 신뢰성을 검증하기 위해, 리뷰 텍스트(`review_text`)를 기반으로 파생 변수를 생성하였습니다.

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

    - 출력 결과
        
        ![데이터 검증 출력 결과](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/data_check.png)

    - 기존 컬럼과 재계산된 값 간의 차이를 비교한 결과, 데이터 전반에서 큰 불일치는 관찰되지 않았으며, 제공된 `review_length`, `num_words` 컬럼은 신뢰 가능한 것으로 판단하였습니다.

- #### 분포 시각화
    주요 수치형 변수의 분포를 확인하기 위해 히스토그램과 박스플롯을 활용하였습니다.

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

    - 출력 결과
        - `review_length`

            ![review_length 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length_histogram.png)
            ![review_length boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/review_length_boxplot.png)

        - `rating`

            ![rating 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating_histogram.png)
            ![rating boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating_boxplot.png)

        - `sentimentic_score`

            ![sentimentic_score 히스토그램](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/sentimentic_score_histogram.png)
            ![rating boxplot](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/sentimentic_score_boxplot.png)


- #### 이상치 탐지 (IQR / Z-score)
    분포 시각화 이후, IQR 및 Z-score 기준을 활용하여 이상치를 정량적으로 탐지하였습니다.
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
    
    - IQR 및 Z-score 기준 모두에서 이상치로 판단되는 데이터는 발견되지 않았고, 데이터 전반이 정상 범위 내에 분포하고 있음을 확인하였습니다. 따라서 이상치 제거 없이 전체 데이터를 분석 대상으로 유지하였습니다.

---

## 3. 기술 통계 및 시각화

### 3.1 주요 변수 기술 통계 요약

| 변수            | 평균    | 중앙값 | 표준편차 |
| --------------- | ------- | ------ | -------- |
| review_length   | 약 124  | 125    | 41.65    |
| num_words       | 약 29   | 28     | 11.31    |
| sentiment_score | 약 0.02 | 0.03   | 0.61     |
| rating          | 약 3.21 | 3      | 1.35     |

---

### 3.2 카테고리별 평균 평점 분석 (barplot)

![카테고리 별 평균 평점](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/category-avg_rating.png)

- `home`, `fashion` 카테고리가 상대적으로 높은 평균 평점을 보임
- 카테고리 간 평균 평점 차이는 크지 않으나 순위 차이는 존재

---

### 3.3 평점과 감성 점수 관계 분석 (violinplot)

![평점과 감성 점수 관계](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/rating-sentiment_score.png)

- 감성 점수가 높아질수록 평점이 함께 증가하는 경향
- 상관계수: **0.73 (강한 양의 상관관계)**

---

### 3.4 텍스트 길이와 평점 관계 분석 (boxplot)

![텍스트 길이와 평점 관계](https://raw.githubusercontent.com/sehaim/skala-python/main/eda_example/output/text_length-avg_rating.png)

- 리뷰 길이와 평점 간 직접적인 관계는 뚜렷하지 않음

---

## 4. AI 분석을 위한 인사이트 도출

### 4.1 감성 점수와 평점의 관계

- 감성 점수 분위수가 증가할수록 평균 평점이 단계적으로 상승
- 감성 점수는 평점 예측에 핵심적인 변수

### 4.2 리뷰 길이와 임베딩 유사도 영향 가능성

- 리뷰 길이는 평점과의 직접 상관은 낮음
- 임베딩 품질 관점에서 적정 길이 범위 설정 필요

### 4.3 카테고리별 감성 점수 차이 분석

- 카테고리별 평균 감성 점수 차이는 존재
- 추천 모델에서 카테고리 정보 활용 필요성 확인

---

## 5. 핵심 인사이트 요약

1. 감성 점수는 평점과 강한 양의 상관관계를 가지며 AI 추천 시스템의 핵심 입력 변수로 활용 가능하다.
2. 리뷰 길이는 평점과 직접적인 관계는 약하지만, 임베딩 품질에 영향을 줄 수 있는 요소로 고려할 필요가 있다.
3. 카테고리별 감성 분포 차이를 반영한 추천 전략이 효과적일 수 있다.

---

## 6. 결론

본 EDA를 통해 리뷰 데이터는 전반적으로 품질이 양호하며, 감성 점수 중심의 AI 임베딩 분석이 충분히 의미 있는 결과를 도출할 수 있음을 확인하였다.  
향후 본 결과를 기반으로 LLM 임베딩 생성 및 유사도 기반 추천 모델을 확장할 수 있다.
