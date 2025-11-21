# coding=utf-8
"""
work_daily_analysis.py

单文件版本 — 可部署到 Streamlit Cloud，实时读取数据库并返回：
    overall_data, weekly_analysis, recent4, duration_analysis

特点：
- 使用 st.secrets，支持 secrets 节点名：db_wms, db_cbs（你在 Cloud 已保存）
- 提供公共函数 get_all_data(db_config=None)
- 使用 @st.cache_data(ttl=300) 做 5 分钟缓存以降低 DB 压力
- 导入模块时会尝试自动用 st.secrets 加载（兼容你原来直接 import overall_data 的用法）
"""

import pymysql
import pandas as pd
import warnings
from datetime import datetime
import numpy as np
import streamlit as st

# ---------- Secrets (read at runtime) ----------
# Ensure you have [db_wms] and [db_cbs] set in Streamlit secrets (Advanced settings)
def _get_db_conf(section: str):
    conf = st.secrets.get(section)
    if conf is None:
        raise RuntimeError(f"Secrets section '{section}' not found. Please set it in Streamlit Advanced settings.")
    return conf
warnings.filterwarnings("ignore")



@st.cache_data(ttl=60)
def _load_and_compute(db_config=None):


    now = pd.Timestamp.now()

    # 查询窗口：最近30天（到昨天结束）
    yesterday_end = (now - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    thirty_days_ago_start = (now - pd.Timedelta(days=30)).normalize()
    yesterday_end_str = yesterday_end.strftime("%Y-%m-%d %H:%M:%S")
    thirty_days_ago_start_str = thirty_days_ago_start.strftime("%Y-%m-%d %H:%M:%S")

    # 连接数据库（使用 db_wms）
    db_wms = _get_db_conf("db_wms")
    conn = pymysql.connect(
        host=db_wms["host"],
        port=int(db_wms.get("port", 3306)),
        user=db_wms["user"],
        password=db_wms["password"],
        database=db_wms["database"],
        charset="utf8"
    )
    # -------------------------
    # 1) 工人考勤
    # -------------------------
    sql_staff = r"""
    SELECT 
        t.staff_name AS 工人名称,
        t.warehouse_code AS 仓库,
        t.occupation AS 工种,
        t.hourly_rate AS 时薪,
        CONCAT(ROUND(t.fee_rate, 0), '%%') AS 中介费率,
        t1.work_date AS 工作日期,
        t1.work_start_time AS 上班打卡时间,
        t1.work_end_time AS 下班打卡时间,
        (SELECT ssr.work_signin_date 
         FROM wms_staff_signin_record ssr 
         WHERE ssr.work_detail_id = t1.id 
           AND ssr.work_signin_type = 'break_start' 
           AND ssr.is_delete = 0 
         LIMIT 1) AS 休息开始时间,
        (SELECT ssr.work_signin_date 
         FROM wms_staff_signin_record ssr 
         WHERE ssr.work_detail_id = t1.id 
           AND ssr.work_signin_type = 'break_end' 
           AND ssr.is_delete = 0 
         LIMIT 1) AS 休息结束时间
    FROM wms_staff_manage t 
    LEFT JOIN wms_staff_work_detail t1 
        ON t.id = t1.staff_id 
       AND t1.is_delete = 0
    WHERE t.is_delete = 0
      AND DATE_FORMAT(t1.work_date, '%%Y-%%m-%%d') >= %s
      AND DATE_FORMAT(t1.work_date, '%%Y-%%m-%%d') <= %s
      AND t.warehouse_code = 'JFK1'
    ORDER BY t.staff_name
    """
    jfk_records = pd.read_sql(sql_staff, conn, params=(thirty_days_ago_start_str, yesterday_end_str))
    if not jfk_records.empty:
        # convert date type and compute columns
        try:
            jfk_records['工作日期'] = pd.to_datetime(jfk_records['工作日期']).dt.date
        except Exception:
            pass

        # 清洗中介费率
        if '中介费率' in jfk_records.columns:
            jfk_records['中介费率'] = pd.to_numeric(
                jfk_records['中介费率'].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            ) / 100
            jfk_records['中介费率'] = jfk_records['中介费率'].fillna(0)

        # 计算工作时长（小时）
        jfk_records['工作时长'] = (
            jfk_records['下班打卡时间'] - jfk_records['上班打卡时间']
            - (jfk_records['休息结束时间'] - jfk_records['休息开始时间'])
        ).dt.total_seconds() / 3600
        jfk_records['当日工资'] = jfk_records['工作时长'] * jfk_records['时薪'] * (1 + jfk_records['中介费率'])
    else:
        # 生成空表结构，避免后续报错
        jfk_records = pd.DataFrame(columns=[
            '工人名称','仓库','工种','时薪','中介费率','工作日期',
            '上班打卡时间','下班打卡时间','休息开始时间','休息结束时间','工作时长','当日工资'
        ])

    jfk_worker_cost = jfk_records.copy()

    # 分工种
    scan_worker = jfk_records[jfk_records['工种'].isin(['Warehouse Labor', 'Warehouse'])]
    fork_lift_operator = jfk_records[jfk_records['工种'].isin(['Fork Lift Operator', 'Forklift Operator'])]

    scan_worker_analysis = scan_worker.groupby('工作日期').agg(
        scan_worker_count=('工人名称', 'count'),
        wages_count1=('当日工资', 'sum'),
        time_count1=('工作时长', 'sum')
    ).reset_index().rename(columns={'工作日期': 'create_date'})
    if not scan_worker_analysis.empty:
        scan_worker_analysis[['wages_count1','time_count1']] = scan_worker_analysis[['wages_count1','time_count1']].round(2)

    fork_lift_operator_analysis = fork_lift_operator.groupby('工作日期').agg(
        fork_lift_operator_count=('工人名称', 'count'),
        wages_count2=('当日工资', 'sum'),
        time_count2=('工作时长', 'sum')
    ).reset_index().rename(columns={'工作日期': 'create_date'})
    if not fork_lift_operator_analysis.empty:
        fork_lift_operator_analysis[['wages_count2','time_count2']] = fork_lift_operator_analysis[['wages_count2','time_count2']].round(2)

    # -------------------------
    # 2) Pallet (收货)
    # -------------------------
    sql_pallet = r"""
    SELECT p.pallet_no, w.warehouse_code, p.create_time, 
    MIN(b.pallet_scan_time) AS firstScanTime, MAX(b.pallet_scan_time) AS lastScanTime, 
    p.PLT_BOX_NUM AS bagCount 
    FROM ifm_warehouse_pallet p
    LEFT JOIN ifm_warehouse_bag b ON b.pallet_id = p.id AND b.mark = 1
    LEFT JOIN sys_warehouse w ON p.warehouse_id = w.id
    WHERE p.mark = 1 
    AND p.create_time BETWEEN %s AND %s
    GROUP BY p.id
    ORDER BY w.warehouse_code;
    """
    pallet_records = pd.read_sql(sql_pallet, conn, params=(thirty_days_ago_start_str, yesterday_end_str))
    pallet_jfk = pallet_records[pallet_records['warehouse_code'] == "JFK1"] if not pallet_records.empty else pd.DataFrame()
    if not pallet_jfk.empty:
        for col in ['create_time', 'firstScanTime', 'lastScanTime']:
            pallet_jfk[col] = pallet_jfk[col] - pd.Timedelta(hours=4)
        pallet_jfk['create_date'] = pallet_jfk['create_time'].dt.date

    # -------------------------
    # 3) Gaylord (投gaylord)
    # -------------------------
    sql_gay = r"""
    SELECT g.gayload_no, w.warehouse_code, g.create_time, 
    MIN(b.gayload_scan_time) AS firstScanTime, MAX(b.gayload_scan_time) AS lastScanTime, 
    g.pieces AS bagCount 
    FROM ifm_warehouse_gayload g
    LEFT JOIN ifm_warehouse_bag b ON b.gayload_id = g.id AND b.mark = 1
    LEFT JOIN sys_warehouse w ON g.warehouse_id = w.id
    WHERE g.mark = 1 
    AND g.create_time BETWEEN %s AND %s
    GROUP BY g.id
    ORDER BY w.warehouse_code;
    """
    gaylord_records = pd.read_sql(sql_gay, conn, params=(thirty_days_ago_start_str, yesterday_end_str))
    gaylord_jfk = gaylord_records[gaylord_records['warehouse_code'] == "JFK1"] if not gaylord_records.empty else pd.DataFrame()
    if not gaylord_jfk.empty:
        for col in ['create_time', 'firstScanTime', 'lastScanTime']:
            gaylord_jfk[col] = gaylord_jfk[col] - pd.Timedelta(hours=4)
        gaylord_jfk['create_date'] = gaylord_jfk['create_time'].dt.date

    # -------------------------
    # 4) SO 出库
    # -------------------------
    sql_so = r"""
    SELECT w.warehouse_code, s.so_no,s.channel,s.line_no, s.departure_time AS outboundTime,
        CASE 
            WHEN s.so_mode = 'EC' THEN MIN(p.loading_time)
            WHEN s.so_mode = 'GAYLOAD' THEN MIN(g.loading_time)
        END AS firstScanTime,
        CASE 
            WHEN s.so_mode = 'EC' THEN MAX(p.loading_time)
            WHEN s.so_mode = 'GAYLOAD' THEN MAX(g.loading_time)
        END AS lastScanTime,
        CASE 
            WHEN s.so_mode = 'EC' THEN s.total_plt_count
            WHEN s.so_mode = 'GAYLOAD' THEN s.total_gayload_count
        END AS containerCount,
        CASE 
            WHEN s.so_mode = 'EC' THEN s.total_plt_box_count
            WHEN s.so_mode = 'GAYLOAD' THEN s.total_bag_count
        END AS totalBagCount 
    FROM so s
    LEFT JOIN ifm_warehouse_pallet p ON p.OUTPLAN_ID = s.id AND p.mark = 1
    LEFT JOIN ifm_warehouse_gayload g ON g.OUTPLAN_ID = s.id AND g.mark = 1
    LEFT JOIN sys_warehouse w ON s.warehouse_id = w.id 
    WHERE s.mark = 1 
    AND s.create_time BETWEEN %s AND %s
    GROUP BY s.id;
    """
    so_records = pd.read_sql(sql_so, conn, params=(thirty_days_ago_start_str, yesterday_end_str))
    # close connection
    conn.close()

    so_jfk = so_records[so_records['warehouse_code'] == "JFK1"] if not so_records.empty else pd.DataFrame()
    if not so_jfk.empty:
        for col in ['outboundTime', 'firstScanTime', 'lastScanTime']:
            so_jfk[col] = so_jfk[col] - pd.Timedelta(hours=4)
        so_jfk['outbound_date'] = so_jfk['outboundTime'].dt.date

    # -------------------------
    # Aggregations and merges (保留你原始计算逻辑)
    # -------------------------
    inbound_analysis = pallet_jfk.groupby('create_date').agg(
        inbound_pallet_count=('pallet_no', 'count'),
        inbound_bag_count=('bagCount', 'sum')
    ).reset_index() if not pallet_jfk.empty else pd.DataFrame(columns=['create_date','inbound_pallet_count','inbound_bag_count'])

    gaylord_analysis = gaylord_jfk.groupby('create_date').agg(
        gaylord_count=('gayload_no', 'count'),
        gaylord_bag_count=('bagCount', 'sum')
    ).reset_index() if not gaylord_jfk.empty else pd.DataFrame(columns=['create_date','gaylord_count','gaylord_bag_count'])

    so_analysis = so_jfk.groupby('outbound_date').agg(
        outbound_so_count=('so_no', 'count'),
        outbound_container_count=('containerCount', 'sum'),
        outbound_bag_count=('totalBagCount', 'sum')
    ).reset_index().rename(columns={'outbound_date':'create_date'}) if not so_jfk.empty else pd.DataFrame(columns=['create_date','outbound_so_count','outbound_container_count','outbound_bag_count'])

    line_analysis = so_jfk.groupby('outbound_date').agg(
        line_count=('line_no','count')
    ).reset_index().rename(columns={'outbound_date':'create_date'}) if not so_jfk.empty else pd.DataFrame(columns=['create_date','line_count'])

    merged = pd.merge(inbound_analysis, gaylord_analysis, on='create_date', how='outer')   # 第一次合并
    merged = pd.merge(merged, so_analysis, on='create_date', how='outer')
    merged = pd.merge(merged, line_analysis, on='create_date', how='outer')

    # ensure expected cols exist
    cols = ['gaylord_count','gaylord_bag_count','outbound_so_count','outbound_container_count','outbound_bag_count','line_count']
    for c in cols:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0)

    merged['inbound_bag_count_actual'] = merged.get('inbound_bag_count', 0).fillna(0) + merged.get('gaylord_bag_count', 0).fillna(0)
    merged['inbound_pallet_count_actual'] = merged.get('inbound_pallet_count', 0).fillna(0) + round(merged.get('gaylord_count', 0).fillna(0) / 1.3, 0)

    # 大包重量参数
    carton_weight_big = 33.75
    carton_weight_small = 21
    big_percent = 0.1
    carton_weight_avg = carton_weight_big * big_percent + carton_weight_small * (1 - big_percent)

    merged['入库重量'] = round((merged.get('inbound_bag_count', 0).fillna(0) + merged.get('gaylord_bag_count', 0).fillna(0)) * carton_weight_avg / 1000, 2)
    merged['出库重量'] = round((merged.get('outbound_bag_count', 0).fillna(0)) * carton_weight_avg / 1000, 2)
    merged['吞吐量'] = merged['入库重量'] + merged['出库重量']

    # merge worker aggregates
    if not scan_worker_analysis.empty:
        merged = merged.merge(scan_worker_analysis, on='create_date', how='outer')
    else:
        # create columns if not present
        for c in ['scan_worker_count','wages_count1','time_count1']:
            if c not in merged.columns:
                merged[c] = 0

    if not fork_lift_operator_analysis.empty:
        merged = merged.merge(fork_lift_operator_analysis, on='create_date', how='outer')
    else:
        for c in ['fork_lift_operator_count','wages_count2','time_count2']:
            if c not in merged.columns:
                merged[c] = 0

    merged.replace([' ', '', 'nan', 'None', 'N/A'], np.nan, inplace=True)
    merged.fillna(0, inplace=True)

    merged['当日工资支出'] = merged['wages_count1'] + merged['wages_count2']
    merged['当日工人数量'] = merged['fork_lift_operator_count'] + merged['scan_worker_count']
    merged['当日工作时长'] = merged['time_count1'] + merged['time_count2']

    # 收货时效工作量估算
    merged['卸车'] = round(merged['inbound_pallet_count_actual'] * 70 / 3600, 2)
    merged['托盘点数'] = round(merged['inbound_pallet_count_actual'] * 30 / 3600, 2)
    merged['标签处理'] = round(merged['inbound_pallet_count_actual'] * 0.15 * 180 / 3600, 2)
    merged['重新打托'] = round(merged['inbound_pallet_count_actual'] * 600 / (3600 * 24), 2)
    merged['拆托'] = round(merged['inbound_pallet_count_actual'] * 120 / (3600 * 6), 2)
    merged['托盘上架'] = round(merged['inbound_pallet_count_actual'] * 45 / 3600, 2)
    merged['上架扫描'] = round(merged['inbound_pallet_count_actual'] * 20 / 3600, 2)
    merged['收货'] = merged[['卸车','托盘点数','标签处理','重新打托','拆托','托盘上架','上架扫描']].sum(axis=1)

    # 出库
    merged['托盘下架'] = round(merged['outbound_container_count'] * 30 / 3600, 2)
    merged['下架扫描'] = round(merged['outbound_container_count'] * 5 / 3600, 2)
    merged['装车'] = round(merged['outbound_container_count'] * 60 / 3600, 2)
    merged['装车扫描'] = round(merged['outbound_container_count'] * 5 / 3600, 2)
    merged['合托'] = round(merged['line_count'] * 150 / 3600, 2)
    merged['出货'] = merged[['托盘下架','下架扫描','装车','装车扫描','合托']].sum(axis=1)

    # 投料
    merged['投gaylord'] = round(merged['gaylord_bag_count'] * 35 / 3600, 2)
    merged['整理空箱'] = round(merged['gaylord_bag_count'] * 15 / 3600, 2)
    merged['补空gaylord'] = round(merged['gaylord_count'] * 60 / 3600, 2)
    merged['缠gaylord'] = round(merged['gaylord_count'] * 45 / 3600, 2)
    merged['投料'] = merged[['投gaylord','整理空箱','补空gaylord','缠gaylord']].sum(axis=1)

    # 日常工作
    merged['日常清理工作'] = 2
    merged['理论工作时长'] = merged[['收货','出货','投料','日常清理工作']].sum(axis=1)

    # 工作饱和度（避免除 0）
    merged['工作饱和度'] = (merged['理论工作时长'] / merged['当日工作时长'].replace({0: np.nan})).round(2).fillna(0)
    merged['叉车工理论工作时长'] = merged['卸车'] + merged['托盘上架'] + merged['托盘下架'] + merged['装车']
    merged['力工理论工作时长'] = merged['理论工作时长'] - merged['叉车工理论工作时长']
    merged['叉车工工作饱和度'] = (merged['叉车工理论工作时长'] / merged['time_count2'].replace({0: np.nan})).round(2).fillna(0)
    merged['力工工作饱和度'] = (merged['力工理论工作时长'] / merged['time_count1'].replace({0: np.nan})).round(2).fillna(0)

    merged['操作成本($/吨)'] = (merged['当日工资支出'] / merged['吞吐量'].replace({0: np.nan})).round(2).fillna(0)
    merged['人效(吨/小时)'] = (merged['吞吐量'] / merged['当日工作时长'].replace({0: np.nan})).round(2).fillna(0)

    # -------------------------
    # 每周工时与成本（基于 jfk_worker_cost）
    # -------------------------
    if not jfk_worker_cost.empty:
        tmp = jfk_worker_cost.copy()
        tmp['week'] = tmp['工作日期'].apply(lambda x: pd.to_datetime(x).isocalendar()[1])
        tmp['year'] = tmp['工作日期'].apply(lambda x: pd.to_datetime(x).isocalendar()[0])
        tmp['Monday'] = tmp['工作日期'].apply(
            lambda x: (pd.to_datetime(x) - pd.to_timedelta(pd.to_datetime(x).weekday(), unit='D')).date()
        )
        latest_week = tmp['week'].max() if not tmp.empty else None
        recent = tmp[(tmp['week'] <= (latest_week if latest_week is not None else 0)) & (tmp['week'] > (latest_week - 5 if latest_week else -999))]
        weekly_hours = recent.groupby(['week','Monday','工人名称'], as_index=False)['工作时长'].sum()
        weekly_hours['工作时长'] = weekly_hours['工作时长'].round(2)
        weekly_hours['正常工时'] = weekly_hours['工作时长'].clip(upper=40).round(2)
        weekly_hours['加班工时'] = (weekly_hours['工作时长'] - 40).clip(lower=0).round(2)
        worker_info = tmp[['工人名称','时薪','中介费率']].drop_duplicates()
        weekly_hours = weekly_hours.merge(worker_info, on='工人名称', how='left')
        weekly_hours['正常工资'] = (weekly_hours['正常工时'] * weekly_hours['时薪'] * (1 + weekly_hours['中介费率'])).round(2)
        weekly_hours['加班工资'] = (weekly_hours['加班工时'] * weekly_hours['时薪'] * 1.5 * (1 + weekly_hours['中介费率'])).round(2)
        weekly_hours['总工资'] = (weekly_hours['正常工资'] + weekly_hours['加班工资']).round(2)
        recent4 = weekly_hours[weekly_hours['week'] > (latest_week - 5 if latest_week else -999)].copy()
        if 'week' in recent4.columns:
            recent4.drop(columns=['week'], inplace=True)
        weekly_summary = recent4.groupby(['Monday']).agg({
            '工作时长':'sum',
            '正常工时':'sum',
            '加班工时':'sum',
            '正常工资':'sum',
            '加班工资':'sum',
            '总工资':'sum'
        }).reset_index()
    else:
        recent4 = pd.DataFrame()
        weekly_summary = pd.DataFrame()

    # -------------------------
    # 每周货量与人力成本合并（基于 merged）
    # -------------------------
    if not merged.empty:
        merged['week'] = merged['create_date'].apply(lambda x: pd.to_datetime(x).isocalendar()[1])
        merged['year'] = merged['create_date'].apply(lambda x: pd.to_datetime(x).isocalendar()[0])
        merged['Monday'] = merged['create_date'].apply(
            lambda x: (pd.to_datetime(x) - pd.to_timedelta(pd.to_datetime(x).weekday(), unit='D')).date()
        )
        latest_week2 = merged['week'].max() if not merged.empty else None
        recent_weight = merged[(merged['week'] <= (latest_week2 if latest_week2 is not None else 0)) & (merged['week'] > (latest_week2 - 5 if latest_week2 else -999))]
        weekly_weight = recent_weight.groupby(['week','Monday'], as_index=False)[['入库重量','出库重量','吞吐量']].sum()
        weekly_weight_ = weekly_weight[['Monday','入库重量','出库重量','吞吐量']]
        if not weekly_summary.empty and not weekly_weight_.empty:
            weekly_analysis = weekly_summary.merge(weekly_weight_, on='Monday', how='inner')
            weekly_analysis['操作成本($/吨)'] = (weekly_analysis['总工资'] / weekly_analysis['吞吐量'].replace({0: np.nan})).round(2).fillna(0)
            weekly_analysis['人效(吨/小时)'] = (weekly_analysis['吞吐量'] / weekly_analysis['工作时长'].replace({0: np.nan})).round(2).fillna(0)
            weekly_analysis = weekly_analysis[['Monday','入库重量','出库重量','吞吐量','工作时长','人效(吨/小时)','总工资','操作成本($/吨)','正常工时','正常工资','加班工时','加班工资']]
        else:
            weekly_analysis = pd.DataFrame()
    else:
        weekly_analysis = pd.DataFrame()

    # -------------------------
    # overall_data & duration_analysis
    # -------------------------
    overall_columns = [
        'create_date','入库重量','出库重量','吞吐量','当日工人数量','当日工作时长',
        '人效(吨/小时)','当日工资支出','操作成本($/吨)','inbound_pallet_count','gaylord_count',
        'line_count','scan_worker_count','wages_count1','fork_lift_operator_count','wages_count2'
    ]
    for c in overall_columns:
        if c not in merged.columns:
            merged[c] = 0

    overall_data = merged[overall_columns].copy()
    overall_data.rename(columns={
        'create_date':'操作日期',
        'scan_worker_count':'力工人数',
        'wages_count1':'力工成本',
        'fork_lift_operator_count':'叉车工人数',
        'wages_count2':'叉车工成本',
        'inbound_pallet_count':'收货托数',
        'gaylord_count':'投Gaylord数',
        'line_count':'发车数'
    }, inplace=True)

    # 去掉今天的数据行
    today = now.normalize()
    overall_data['操作日期'] = pd.to_datetime(overall_data['操作日期'], errors='coerce')
    overall_data = overall_data[overall_data['操作日期'].dt.normalize() != today]
    overall_data['操作日期'] = overall_data['操作日期'].dt.date

    duration_analysis_cols = [
        'create_date','吞吐量','当日工人数量','当日工作时长','理论工作时长','工作饱和度',
        'scan_worker_count','time_count1','力工理论工作时长','力工工作饱和度',
        'fork_lift_operator_count','time_count2','叉车工理论工作时长','叉车工工作饱和度'
    ]
    for c in duration_analysis_cols:
        if c not in merged.columns:
            merged[c] = 0
    duration_analysis = merged[duration_analysis_cols].copy()
    duration_analysis.rename(columns={
        'create_date':'操作日期',
        'scan_worker_count':'力工人数',
        'time_count1':'力工工时',
        'fork_lift_operator_count':'叉车工人数',
        'time_count2':'叉车工工时'
    }, inplace=True)
    duration_analysis['操作日期'] = pd.to_datetime(duration_analysis['操作日期'], errors='coerce')
    duration_analysis = duration_analysis[duration_analysis['操作日期'].dt.normalize() != today]
    duration_analysis['操作日期'] = duration_analysis['操作日期'].dt.date

    # 返回最终四张表
    return overall_data, weekly_analysis, recent4, duration_analysis


def get_all_data(db_config=None):
    """
    公共接口：返回 overall_data, weekly_analysis, recent4, duration_analysis
    db_config: 可选 dict，若传入则使用；否则使用 st.secrets['db_wms']
    """
    return _load_and_compute(db_config)

