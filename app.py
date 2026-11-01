import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='국내 질병부담과 임상연구', layout='wide')
st.title('국내 질병부담과 임상연구 분포 분석')
st.caption('대한민국 · 2010–2023 · GBD Level 2 17개 질병군 · CRIS 중재연구')

@st.cache_data
def load():
    a = pd.read_csv('data/최종_17개질병군.csv')
    b = pd.read_csv('data/CRIS_17개질병군_웹앱용.csv')
    return a,b

df,cris=load()
page=st.sidebar.radio('분석 페이지',['1. 질병부담과 임상연구 현황','2. 질병부담과 임상연구의 불균형'])
years=st.sidebar.slider('분석 기간',(int(df.year.min())),(int(df.year.max())),(2010,2023))
selected=df[df.year.between(*years)]
agg=selected.groupby('cause_name',as_index=False).agg(daly=('daly','sum'),research_count=('research_count','sum'),weighted_research_count=('weighted_research_count','sum'))
agg['daly_burden_share']=agg.daly/agg.daly.sum()
total=agg.weighted_research_count.sum()
agg['research_share']=agg.weighted_research_count/total if total else 0.0
agg['share_gap']=agg.daly_burden_share-agg.research_share
agg['burden_research_ratio']=agg.daly_burden_share/agg.research_share.replace(0,float('nan'))
agg=agg.sort_values('daly_burden_share',ascending=False)

st.info('CRIS 질병군 분류는 원문 검증이 완료되지 않은 잠정 결과입니다. 등록 연구 건수는 연구의 질·규모·전체 연구투자를 의미하지 않습니다.')
if page.startswith('1.'):
    st.subheader('질병부담과 임상연구 현황')
    st.caption('히트맵은 기간 선택과 관계없이 2010~2023년 전체를 표시합니다.')
    pivot=df.pivot(index='cause_name',columns='year',values='daly')
    fig=px.imshow(pivot,aspect='auto',labels={'color':'DALY','x':'연도','y':'질병군'},title='질병군별 DALY 히트맵 (2010–2023 고정)')
    st.plotly_chart(fig,use_container_width=True)
    metric=st.radio('점유율 선택',['DALY 점유율','임상연구 점유율'],horizontal=True)
    col='daly_burden_share' if metric=='DALY 점유율' else 'research_share'
    st.plotly_chart(px.bar(agg,x='cause_name',y=col,title=f'{years[0]}–{years[1]} {metric}',labels={col:'점유율', 'cause_name':'질병군'}),use_container_width=True)
else:
    st.subheader('질병부담과 임상연구의 불균형')
    fig=px.scatter(agg,x='research_share',y='daly_burden_share',hover_name='cause_name',size='daly',title='임상연구 점유율 vs DALY 점유율',labels={'research_share':'임상연구 점유율','daly_burden_share':'DALY 점유율'})
    fig.add_shape(type='line',x0=0,y0=0,x1=max(agg.research_share.max(),agg.daly_burden_share.max()),y1=max(agg.research_share.max(),agg.daly_burden_share.max()),line={'dash':'dash','color':'gray'})
    st.plotly_chart(fig,use_container_width=True)
    long=agg.melt(id_vars='cause_name',value_vars=['daly_burden_share','research_share'],var_name='지표',value_name='점유율')
    st.plotly_chart(px.bar(long,x='cause_name',y='점유율',color='지표',barmode='group',title='질병군별 점유율 비교'),use_container_width=True)
    st.caption('불균형율 = DALY 점유율 ÷ 임상연구 점유율. 연구 점유율이 0이면 비율은 정의되지 않습니다.')
    st.dataframe(agg[['cause_name','daly_burden_share','research_share','burden_research_ratio','share_gap']].rename(columns={'cause_name':'질병군','daly_burden_share':'DALY 점유율','research_share':'임상연구 점유율','burden_research_ratio':'불균형율','share_gap':'점유율 차이'}),hide_index=True,use_container_width=True)

with st.expander('CRIS 연구 분류 현황'):
    st.write(f'원본 CRIS 연구 {len(cris):,}건')
    st.dataframe(cris['updated_classification_status'].fillna('상태 없음').value_counts().rename_axis('분류 상태').reset_index(name='건수'),hide_index=True)
    st.dataframe(cris[['trial_id','year','scientific_title_kr','updated_17_cause_name','updated_classification_status']].head(100),hide_index=True)

st.download_button('현재 기간 분석표 CSV 다운로드',agg.to_csv(index=False).encode('utf-8-sig'),file_name='분석결과.csv',mime='text/csv')
