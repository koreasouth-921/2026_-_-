from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="국내 질병부담과 임상연구 분포 분석",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent


# CSV 파일 자동 검색
def normalized(text):
    return unicodedata.normalize(
        "NFC", str(text)
    ).casefold()


def find_csv(candidates, prefix):
    all_files = []

    for folder in [ROOT, ROOT / "data"]:
        if folder.exists():
            all_files.extend([
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".csv"
            ])

    by_name = {
        normalized(p.name): p
        for p in all_files
    }

    for candidate in candidates:
        key = normalized(candidate)

        if key in by_name:
            return by_name[key]

    matches = [
        p for p in all_files
        if normalized(p.name).startswith(
            normalized(prefix)
        )
    ]

    if len(matches) == 1:
        return matches[0]

    st.error("필요한 CSV 파일을 찾지 못했습니다.")
    st.write(
        "현재 발견된 CSV 파일:",
        [p.name for p in all_files]
    )
    st.stop()


# 데이터 불러오기
@st.cache_data
def load_data(final_path, cris_path):
    final = pd.read_csv(
        final_path,
        encoding="utf-8-sig"
    )

    required_cris = [
        "trial_id",
        "year",
        "scientific_title_kr",
        "updated_17_cause_name",
        "updated_classification_status"
    ]

    header = pd.read_csv(
        cris_path,
        nrows=0
    ).columns.tolist()

    available = [
        col for col in required_cris
        if col in header
    ]

    cris = pd.read_csv(
        cris_path,
        usecols=available,
        encoding="utf-8-sig",
        low_memory=False
    )

    return final, cris


final_file = find_csv(
    ["최종_17개질병군.csv"],
    "최종_17개질병군"
)

cris_file = find_csv(
    [
        "CRIS_17개질병군.csv",
        "CRIS_17개질병군_웹앱용.csv"
    ],
    "CRIS_17개질병군"
)

df, cris = load_data(
    str(final_file),
    str(cris_file)
)


# 데이터 검증 및 전처리
required = [
    "year",
    "cause_name",
    "daly",
    "research_count",
    "weighted_research_count"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    st.error(
        f"최종 데이터에 필요한 열이 없습니다: {missing}"
    )
    st.write("현재 열:", df.columns.tolist())
    st.stop()

for col in [
    "year",
    "daly",
    "research_count",
    "weighted_research_count"
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=["year", "cause_name"]
).copy()

df["year"] = df["year"].astype(int)

df["cause_name"] = (
    df["cause_name"]
    .astype(str)
    .str.strip()
)

for col in [
    "daly",
    "research_count",
    "weighted_research_count"
]:
    df[col] = df[col].fillna(0)


# 웹앱 제목
st.title("국내 질병부담과 임상연구 분포 분석")

st.caption(
    "대한민국 · 2010–2023 · "
    "GBD Level 2 · 17개 질병군 · CRIS 중재연구"
)

st.warning(
    "CRIS 질병군 분류는 원문 검증이 완료되지 않은 "
    "잠정 결과입니다. 연구 건수는 연구비, "
    "연구의 질이나 실제 임상시험 규모를 "
    "의미하지 않습니다."
)


# 사이드바
page = st.sidebar.radio(
    "분석 페이지",
    [
        "1. 질병부담과 임상연구 현황",
        "2. 질병부담과 임상연구의 불균형"
    ]
)

min_year = int(df["year"].min())
max_year = int(df["year"].max())

start, end = st.sidebar.slider(
    "분석 기간",
    min_year,
    max_year,
    (min_year, max_year)
)


# 선택 기간 집계
selected = df.loc[
    df["year"].between(start, end)
]

agg = selected.groupby(
    "cause_name",
    as_index=False
).agg(
    daly=("daly", "sum"),
    research_count=("research_count", "sum"),
    weighted_research_count=(
        "weighted_research_count", "sum"
    )
)

daly_total = agg["daly"].sum()

research_total = (
    agg["weighted_research_count"].sum()
)

agg["daly_burden_share"] = (
    agg["daly"] / daly_total
    if daly_total else 0.0
)

agg["research_share"] = (
    agg["weighted_research_count"] / research_total
    if research_total else 0.0
)

agg["share_gap"] = (
    agg["daly_burden_share"]
    - agg["research_share"]
)

agg["burden_research_ratio"] = (
    agg["daly_burden_share"]
    / agg["research_share"].replace(0, np.nan)
)

agg = agg.sort_values(
    "daly_burden_share",
    ascending=False
)


# 핵심 지표
c1, c2, c3 = st.columns(3)

c1.metric(
    "분석 질병군",
    f'{agg["cause_name"].nunique()}개'
)

c2.metric(
    "선택 기간 DALY 합계",
    f"{daly_total:,.0f}"
)

c3.metric(
    "가중 임상연구 건수 합계",
    f"{research_total:,.1f}건"
)


# 페이지 1: 현황
if page.startswith("1."):

    st.subheader(
        "1. 질병부담과 임상연구 현황"
    )

    st.caption(
        "히트맵은 분석 기간 선택과 관계없이 "
        "2010~2023년 전체를 표시합니다."
    )

    pivot = df.pivot_table(
        index="cause_name",
        columns="year",
        values="daly",
        aggfunc="sum"
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        labels={
            "x": "연도",
            "y": "질병군",
            "color": "DALY"
        },
        title="질병군별 DALY 히트맵 (2010–2023)"
    )

    fig.update_layout(height=650)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    metric = st.radio(
        "비교할 지표",
        [
            "DALY 점유율",
            "임상연구 점유율"
        ],
        horizontal=True
    )

    col = (
        "daly_burden_share"
        if metric == "DALY 점유율"
        else "research_share"
    )

    fig = px.bar(
        agg,
        x="cause_name",
        y=col,
        title=f"{start}–{end}년 {metric}",
        labels={
            "cause_name": "질병군",
            col: "점유율"
        }
    )

    fig.update_yaxes(
        tickformat=".1%"
    )

    fig.update_layout(
        xaxis_tickangle=-40,
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.pie(
        agg,
        names="cause_name",
        values=col,
        title=f"{metric} 구성비"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# 페이지 2: 불균형
else:

    st.subheader(
        "2. 질병부담과 임상연구의 불균형"
    )

    st.caption(
        "점선 위에 위치한 질병군은 "
        "DALY 점유율이 임상연구 점유율보다 큽니다."
    )

    fig = px.scatter(
        agg,
        x="research_share",
        y="daly_burden_share",
        hover_name="cause_name",
        size="daly",
        labels={
            "research_share": "임상연구 점유율",
            "daly_burden_share": "DALY 점유율"
        },
        title="질병부담과 임상연구 점유율 비교"
    )

    limit = float(
        max(
            agg["research_share"].max(),
            agg["daly_burden_share"].max(),
            0.01
        )
    ) * 1.05

    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            line=dict(
                color="gray",
                dash="dash"
            ),
            name="점유율 동일선"
        )
    )

    fig.update_xaxes(
        range=[0, limit],
        tickformat=".1%"
    )

    fig.update_yaxes(
        range=[0, limit],
        tickformat=".1%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    long = agg.melt(
        id_vars="cause_name",
        value_vars=[
            "daly_burden_share",
            "research_share"
        ],
        var_name="지표",
        value_name="점유율"
    )

    long["지표"] = long["지표"].replace({
        "daly_burden_share": "DALY 점유율",
        "research_share": "임상연구 점유율"
    })

    fig = px.bar(
        long,
        x="cause_name",
        y="점유율",
        color="지표",
        barmode="group",
        title="질병군별 점유율 비교"
    )

    fig.update_yaxes(
        tickformat=".1%"
    )

    fig.update_layout(
        xaxis_tickangle=-40,
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "불균형율 = DALY 점유율 ÷ 임상연구 점유율. "
        "임상연구 점유율이 0이면 정의되지 않습니다."
    )

    display = agg[
        [
            "cause_name",
            "daly_burden_share",
            "research_share",
            "burden_research_ratio",
            "share_gap"
        ]
    ].rename(
        columns={
            "cause_name": "질병군",
            "daly_burden_share": "DALY 점유율",
            "research_share": "임상연구 점유율",
            "burden_research_ratio": "불균형율",
            "share_gap": "점유율 차이"
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True
    )


# CRIS 분류 현황
with st.expander("CRIS 연구 분류 현황"):

    st.write(
        f"CRIS 연구 데이터: {len(cris):,}건"
    )

    if "updated_classification_status" in cris.columns:

        counts = (
            cris["updated_classification_status"]
            .fillna("상태 없음")
            .value_counts()
            .rename_axis("분류 상태")
            .reset_index(name="건수")
        )

        st.dataframe(
            counts,
            hide_index=True,
            use_container_width=True
        )

    cols = [
        c for c in [
            "trial_id",
            "year",
            "scientific_title_kr",
            "updated_17_cause_name",
            "updated_classification_status"
        ]
        if c in cris.columns
    ]

    st.dataframe(
        cris[cols].head(100),
        hide_index=True,
        use_container_width=True
    )


# 분석 결과 다운로드
st.download_button(
    "선택 기간 분석 결과 CSV 다운로드",
    agg.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=(
        f"질병부담_임상연구_{start}_{end}.csv"
    ),
    mime="text/csv"
)
