import streamlit as st
from work_daily_analysis import get_all_data
from mirage_analysis import get_daily_summary_ata

st.set_page_config(page_title="JFK运营报表", layout="wide")
st.title("JFK运营报表")

daily_summary_ata = get_daily_summary_ata()

# 自动加载最新数据（已带缓存和自动刷新）
overall_data, weekly_analysis, recent4, duration_analysis = get_all_data()

daily_summary_ata_=daily_summary_ata.sort_values(by='ata_date',ascending=False).head(10)
daily_summary_ata_.rename(columns={'延误-严重':'延误-严重（超过24小时）'},inplace=True)

overall_data_=overall_data.sort_values(by='操作日期',ascending=False)
weekly_analysis_=weekly_analysis.sort_values(by='Monday',ascending=False)
recent4_=recent4.sort_values(by='Monday',ascending=False)
duration_analysis_=duration_analysis.sort_values(by='操作日期',ascending=False)

st.expander("一、MIRA时效").dataframe(daily_summary_ata_)
st.markdown("---")

st.expander("二、每天操作量和成本").dataframe(overall_data_)
st.markdown("---")

st.expander("三、每周操作量和成本").dataframe(weekly_analysis_)
st.markdown("---")

st.expander("四、每周工人成本统计").dataframe(recent4_)
st.markdown("---")

st.expander("五、工作饱和度估算").dataframe(duration_analysis_)
