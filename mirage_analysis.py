# coding=utf-8
import pandas as pd
import pymysql
import warnings
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

# ---------- Secrets (read at runtime) ----------
# Ensure you have [db_wms] and [db_cbs] set in Streamlit secrets (Advanced settings)
def _get_db_conf(section: str):
    conf = st.secrets.get(section)
    if conf is None:
        raise RuntimeError(f"Secrets section '{section}' not found. Please set it in Streamlit Advanced settings.")
    return conf

warnings.filterwarnings("ignore")  # 忽略警告


def get_daily_summary_ata():
    """
    获取 JFK 仓库每日时效分析数据
    db_config: dict, 需要包含 host, port, user, password, database
    """
    now = pd.Timestamp.now()
    yesterday_end = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    fifteen_days_ago_start = (now - pd.Timedelta(days=15)).normalize()

    yesterday_end_str = yesterday_end.strftime("%Y-%m-%d %H:%M:%S")
    fifteen_days_ago_start_str = fifteen_days_ago_start.strftime("%Y-%m-%d %H:%M:%S")

    # 查询主单数据
    db_cbs = _get_db_conf("db_cbs")
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs.get("port", 3306)),
        user=db_cbs["user"],
        password=db_cbs["password"],
        database=db_cbs["database"],
        charset="utf8"
    )
    sql_ = """
    SELECT
        a.*,
        c.delivery_channel,
        IF(a.outbound IS NULL, 'N', 'Y') outbound_status,
        CASE a.pod_code
            WHEN 'LAX' THEN DATE_SUB(a.ata, INTERVAL 7 HOUR)
            WHEN 'JFK' THEN DATE_SUB(a.ata, INTERVAL 4 HOUR)
            ELSE DATE_SUB(a.ata, INTERVAL 5 HOUR)
        END AS ata_local,
        CASE a.pod_code
            WHEN 'LAX' THEN DATE_SUB(a.full_release, INTERVAL 7 HOUR)
            WHEN 'JFK' THEN DATE_SUB(a.full_release, INTERVAL 4 HOUR)
            ELSE DATE_SUB(a.full_release, INTERVAL 5 HOUR)
        END AS full_release_local,
        CASE a.pod_code
            WHEN 'LAX' THEN DATE_SUB(a.cbp_release, INTERVAL 7 HOUR)
            WHEN 'JFK' THEN DATE_SUB(a.cbp_release, INTERVAL 4 HOUR)
            ELSE DATE_SUB(a.cbp_release, INTERVAL 5 HOUR)
        END AS cbp_release_local,
        CASE a.pod_code
            WHEN 'LAX' THEN DATE_SUB(a.pga_release, INTERVAL 7 HOUR)
            WHEN 'JFK' THEN DATE_SUB(a.pga_release, INTERVAL 4 HOUR)
            ELSE DATE_SUB(a.pga_release, INTERVAL 5 HOUR)
        END AS pga_release_local
    FROM
        (
        SELECT
            o.id,
            e.mawb_no,
            o.customer_code,
            e.is_destination_transfer scf_type,
            e.transfer_status,
            o.pod_code,
            e.etd,
            e.item_no,
            (SELECT ev.operate_date FROM tl_order_event ev 
             WHERE ev.order_id = o.id AND ev.event_type_id = 18 AND ev.is_delete = 0 LIMIT 1) ata,
            (SELECT ev.operate_date FROM tl_order_event ev 
             WHERE ev.order_id = o.id AND ev.event_type_id = 10 AND ev.is_delete = 0 LIMIT 1) full_release,
            (SELECT ev.operate_date FROM tl_order_event ev 
             WHERE ev.order_id = o.id AND ev.event_type_id = 57 AND ev.is_delete = 0 LIMIT 1) cbp_release,
            (SELECT ev.operate_date FROM tl_order_event ev 
             WHERE ev.order_id = o.id AND ev.event_type_id = 59 AND ev.is_delete = 0 LIMIT 1) pga_release,
            (SELECT ev.operate_date FROM tl_order_event ev 
             WHERE ev.order_id = o.id AND ev.event_type_id = 16 AND ev.is_delete = 0 AND ev.remark IS NULL LIMIT 1) outbound
        FROM tl_order o, tl_order_extra e
        WHERE o.create_time >= %s AND o.create_time <= %s
            AND o.is_delete = 0 AND o.id = e.order_id
            AND e.business_type = '17' AND e.is_delete = 0
        ) a,
        tl_customers_delivery_channel c
    WHERE a.id = c.order_id AND c.is_delete = 0;
    """
    no_outbound_mawb = pd.read_sql(sql_, conn, params=(fifteen_days_ago_start_str, yesterday_end_str))
    conn.close()

    # 过滤 JFK
    jfk_mawb = (no_outbound_mawb['pod_code'] == 'JFK') & (no_outbound_mawb['ata'].notna())
    jfk_mawb_df = no_outbound_mawb.loc[jfk_mawb]
    jfk_mawb_df_ = jfk_mawb_df[['mawb_no', 'item_no', 'ata', 'cbp_release', 'pga_release']]
    jfk_mawb_df_.rename(columns={'ata': 'ata_cbs'}, inplace=True)

    jfk = jfk_mawb_df['mawb_no'].unique().tolist()
    if not jfk:
        return pd.DataFrame()  # 如果没有数据，返回空

    placeholders = ",".join(["%s"] * len(jfk))

    # 查询大包扫描时间
    db_cbs = _get_db_conf("db_wms")
    conn = pymysql.connect(
        host=db_cbs["host"],
        port=int(db_cbs.get("port", 3306)),
        user=db_cbs["user"],
        password=db_cbs["password"],
        database=db_cbs["database"],
        charset="utf8"
    )
    sql_ = f"""
    SELECT m.mawb_no,b.bag_no,m.ata,b.channel,m.photo_upload_date AS channelInTime,
           b.warehouse_out_time,b.pallet_scan_time,b.gayload_scan_time 
    FROM ifm_warehouse_bag b 
    LEFT JOIN ifm_warehouse_mawb m ON b.warehouse_mawb_id = m.id AND m.mark = 1
    WHERE m.mawb_no IN ({placeholders});
    """
    carton_scan_time = pd.read_sql(sql_, conn, params=jfk)
    conn.close()

    carton_scan_time_ = carton_scan_time[carton_scan_time['channel'] != 'ACI']
    carton_scan_time_ = carton_scan_time_[carton_scan_time_['pallet_scan_time'].notnull()]

    # 每个主单最早的时间
    earliest = carton_scan_time_.sort_values('pallet_scan_time').groupby('mawb_no', as_index=False).first()

    merged = pd.merge(earliest, jfk_mawb_df_, on='mawb_no', how='inner')

    columns = ['ata_cbs', 'cbp_release', 'channelInTime', 'warehouse_out_time', 'pallet_scan_time', 'gayload_scan_time']
    merged[columns] = merged[columns].apply(lambda r: r - pd.Timedelta(hours=4))

    # 调整时间函数
    def adjust_time(t):
        if pd.isna(t):
            return t
        if t.hour >= 15:
            return (t + pd.Timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        elif t.hour < 9:
            return t.replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            return t

    merged['ata_reset'] = merged['ata_cbs'].apply(adjust_time)
    merged['start'] = merged[['ata_reset', 'cbp_release']].max(axis=1)
    merged['kpi'] = ((merged['pallet_scan_time'] - merged['start']) < pd.Timedelta(hours=12))
    merged['gap'] = (merged['pallet_scan_time'] - merged['start']).dt.total_seconds() / 3600
    merged.drop_duplicates(inplace=True)

    def grade(r):
        if r <= 12:
            return "达标"
        elif r <= 24:
            return "延误-可控"
        else:
            return "延迟-严重"

    merged["grade"] = merged["gap"].apply(grade)
    merged['ata_date'] = merged['ata_reset'].dt.date

    # 生成每日汇总表
    daily_summary_ata = pd.pivot_table(
        merged,
        index='ata_date',
        columns='grade',
        values='mawb_no',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # 计算总量与严重延误率
    for col in ["达标", "延误-可控", "延迟-严重"]:
        if col not in daily_summary_ata:
            daily_summary_ata[col] = 0
    daily_summary_ata['total'] = daily_summary_ata[['达标', '延误-可控', '延迟-严重']].sum(axis=1)
    daily_summary_ata['严重延误率'] = (daily_summary_ata['延迟-严重'] / daily_summary_ata['total']).round(2)
    daily_summary_ata = daily_summary_ata[['ata_date', '达标', '延误-可控', '延迟-严重', 'total', '严重延误率']]

    return daily_summary_ata
