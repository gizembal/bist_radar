import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────
BIST50_HISSELER = [
    'AKBNK.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
    'TSKB.IS', 'VAKBN.IS', 'YKBNK.IS',
    'BRSAN.IS', 'EREGL.IS', 'KRDMD.IS',
    'GUBRF.IS', 'HEKTS.IS', 'PETKM.IS', 'SASA.IS',
    'SISE.IS', 'TUPRS.IS',
    'ALARK.IS', 'DOHOL.IS', 'KCHOL.IS', 'SAHOL.IS',
    'PASEU.IS', 'PGSUS.IS', 'TAVHL.IS', 'THYAO.IS',
    'BIMAS.IS', 'CCOLA.IS', 'MGROS.IS', 'SOKM.IS', 'ULKER.IS',
    'DOAS.IS', 'FROTO.IS', 'TOASO.IS',
    'BTCIM.IS', 'CIMSA.IS', 'OYAKC.IS',
    'ASTOR.IS', 'KONTR.IS',
    'TCELL.IS', 'TTKOM.IS',
    'ASELS.IS', 'ENKAI.IS', 'EKGYO.IS',
    'KUYAS.IS', 'AEFES.IS', 'MAVI.IS',
    'ARCLK.IS', 'MIATK.IS', 'DSTKF.IS',
]

# ─────────────────────────────────────
# TEKNİK HESAPLAMA
# ─────────────────────────────────────
def teknik_hesapla(df):
    sonuc = []
    for hisse in df['ticker'].unique():
        d = df[df['ticker'] == hisse].copy()

        d['fiyat_degisim'] = d['kapanis'].pct_change() * 100
        d['hacim_oran'] = d['hacim'] / d['hacim'].rolling(20).mean()
        d['volatilite'] = (d['yuksek'] - d['dusuk']) / d['kapanis'] * 100

        # RSI
        delta = d['kapanis'].diff()
        kazan = delta.clip(lower=0).rolling(14).mean()
        kayip = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI'] = (100 - (100 / (1 + kazan / kayip))).round(2)
        d['RSI_sinyal'] = 'Nötr'
        d.loc[d['RSI'] > 80, 'RSI_sinyal'] = 'Çok Aşırı Alınmış'
        d.loc[(d['RSI'] > 70) & (d['RSI'] <= 80), 'RSI_sinyal'] = 'Aşırı Alınmış'
        d.loc[(d['RSI'] < 30) & (d['RSI'] >= 20), 'RSI_sinyal'] = 'Aşırı Satılmış'
        d.loc[d['RSI'] < 20, 'RSI_sinyal'] = 'Çok Aşırı Satılmış'

        # MA
        d['MA20'] = d['kapanis'].rolling(20).mean().round(2)
        d['MA50'] = d['kapanis'].rolling(50).mean().round(2)
        d['MA200'] = d['kapanis'].rolling(200).mean().round(2)
        d['MA_sinyal'] = 'Nötr'
        d.loc[(d['kapanis'] > d['MA50']) & (d['MA50'] > d['MA200']), 'MA_sinyal'] = 'Güçlü Yükseliş Trendi'
        d.loc[(d['kapanis'] < d['MA50']) & (d['MA50'] < d['MA200']), 'MA_sinyal'] = 'Güçlü Düşüş Trendi'
        d.loc[(d['kapanis'] > d['MA50']) & (d['MA50'] < d['MA200']), 'MA_sinyal'] = 'Toparlanma'
        d.loc[(d['kapanis'] < d['MA50']) & (d['MA50'] > d['MA200']), 'MA_sinyal'] = 'Zayıflama'

        cross_once = d['MA50'].shift(1) > d['MA200'].shift(1)
        cross_bugun = d['MA50'] > d['MA200']
        d['Cross_sinyal'] = 'Yok'
        d.loc[(~cross_once) & cross_bugun, 'Cross_sinyal'] = '🌟 Golden Cross'
        d.loc[cross_once & (~cross_bugun), 'Cross_sinyal'] = '💀 Death Cross'

        # MACD
        ema12 = d['kapanis'].ewm(span=12, adjust=False).mean()
        ema26 = d['kapanis'].ewm(span=26, adjust=False).mean()
        d['MACD'] = (ema12 - ema26).round(3)
        d['MACD_sinyal_cizgi'] = d['MACD'].ewm(span=9, adjust=False).mean().round(3)
        d['MACD_histogram'] = (d['MACD'] - d['MACD_sinyal_cizgi']).round(3)
        d['MACD_sinyal'] = 'Nötr'
        d.loc[(d['MACD'] > 0) & (d['MACD_histogram'] > 0), 'MACD_sinyal'] = 'Güçlü Yükseliş'
        d.loc[(d['MACD'] > 0) & (d['MACD_histogram'] < 0), 'MACD_sinyal'] = 'Yükseliş Zayıflıyor'
        d.loc[(d['MACD'] < 0) & (d['MACD_histogram'] > 0), 'MACD_sinyal'] = 'Düşüş Zayıflıyor'
        d.loc[(d['MACD'] < 0) & (d['MACD_histogram'] < 0), 'MACD_sinyal'] = 'Güçlü Düşüş'

        # Bollinger
        bb_ort = d['kapanis'].rolling(20).mean()
        bb_std = d['kapanis'].rolling(20).std()
        d['BB_upper'] = (bb_ort + 2 * bb_std).round(2)
        d['BB_lower'] = (bb_ort - 2 * bb_std).round(2)
        d['BB_middle'] = bb_ort.round(2)
        d['BB_pozisyon'] = ((d['kapanis'] - d['BB_lower']) / (d['BB_upper'] - d['BB_lower'])).round(3)
        d['BB_genislik'] = ((d['BB_upper'] - d['BB_lower']) / d['BB_middle'] * 100).round(2)
        d['BB_genislik_ort'] = d['BB_genislik'].rolling(20).mean()
        d['BB_sinyal'] = 'Normal Bölge'
        d.loc[d['BB_pozisyon'] > 1.0, 'BB_sinyal'] = 'Üst Kırılım'
        d.loc[(d['BB_pozisyon'] > 0.8) & (d['BB_pozisyon'] <= 1.0), 'BB_sinyal'] = 'Üst Bölge'
        d.loc[d['BB_pozisyon'] < 0.0, 'BB_sinyal'] = 'Alt Kırılım'
        d.loc[(d['BB_pozisyon'] < 0.2) & (d['BB_pozisyon'] >= 0.0), 'BB_sinyal'] = 'Alt Bölge'
        d.loc[d['BB_genislik'] < d['BB_genislik_ort'] * 0.7, 'BB_sinyal'] = 'BB Squeeze'

        # OBV
        d['OBV'] = (np.sign(d['kapanis'].diff()) * d['hacim']).cumsum()
        d['OBV_MA20'] = d['OBV'].rolling(20).mean()
        d['OBV_sinyal'] = 'Nötr'
        d.loc[d['OBV'] > d['OBV_MA20'], 'OBV_sinyal'] = 'Alım Baskısı'
        d.loc[d['OBV'] < d['OBV_MA20'], 'OBV_sinyal'] = 'Satış Baskısı'

        # Divergence
        for pencere in [10, 20]:
            fiyat_deg = d['kapanis'].pct_change(pencere) * 100
            rsi_deg = d['RSI'].diff(pencere)
            kolon = f'Div_{pencere}g'
            d[kolon] = 'Yok'
            d.loc[(fiyat_deg < -3) & (rsi_deg > 5) & (d['RSI'] < 50), kolon] = '📈 Bullish'
            d.loc[(fiyat_deg > 3) & (rsi_deg < -5) & (d['RSI'] > 50), kolon] = '📉 Bearish'

        # Güç Skoru
        def hesapla_skor(row):
            skor = 0
            yukselis, dusus = [], []
            if row['RSI_sinyal'] in ['Aşırı Satılmış', 'Çok Aşırı Satılmış']:
                yukselis.append('RSI'); skor += 1
            elif row['RSI_sinyal'] in ['Aşırı Alınmış', 'Çok Aşırı Alınmış']:
                dusus.append('RSI'); skor += 1
            if row['MA_sinyal'] == 'Güçlü Yükseliş Trendi':
                yukselis.append('MA'); skor += 1
            elif row['MA_sinyal'] == 'Güçlü Düşüş Trendi':
                dusus.append('MA'); skor += 1
            if row['MACD_sinyal'] == 'Güçlü Yükseliş':
                yukselis.append('MACD'); skor += 1
            elif row['MACD_sinyal'] == 'Güçlü Düşüş':
                dusus.append('MACD'); skor += 1
            if row['BB_sinyal'] == 'Alt Kırılım':
                yukselis.append('BB'); skor += 1
            elif row['BB_sinyal'] == 'Üst Kırılım':
                dusus.append('BB'); skor += 1
            if row['OBV_sinyal'] == 'Alım Baskısı':
                yukselis.append('OBV'); skor += 1
            elif row['OBV_sinyal'] == 'Satış Baskısı':
                dusus.append('OBV'); skor += 1
            if row['Div_10g'] == '📈 Bullish':
                yukselis.append('DIV10'); skor += 1
            elif row['Div_10g'] == '📉 Bearish':
                dusus.append('DIV10'); skor += 1
            if row['Div_20g'] == '📈 Bullish':
                yukselis.append('DIV20'); skor += 2
            elif row['Div_20g'] == '📉 Bearish':
                dusus.append('DIV20'); skor += 2
            if row['Cross_sinyal'] == '🌟 Golden Cross':
                yukselis.append('CROSS'); skor += 2
            elif row['Cross_sinyal'] == '💀 Death Cross':
                dusus.append('CROSS'); skor += 2

            genel_yon = '📈 Yükseliş' if len(yukselis) > len(dusus) else \
                        '📉 Düşüş' if len(dusus) > len(yukselis) else '➡️ Nötr'
            seviye = '🔴 Çok Güçlü' if skor >= 6 else \
                     '🟠 Güçlü' if skor >= 4 else \
                     '🟡 Orta' if skor >= 2 else \
                     '🔵 Zayıf' if skor == 1 else '✅ Normal'

            return pd.Series({
                'guc_skoru': skor,
                'sinyal_seviyesi': seviye,
                'genel_yon': genel_yon,
                'aktif_sinyaller': ', '.join(yukselis + dusus)
            })

        skorlar = d.apply(hesapla_skor, axis=1)
        d = pd.concat([d, skorlar], axis=1)
        sonuc.append(d)

    df_final = pd.concat(sonuc, ignore_index=True)
    return df_final.replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────
# VERİ ÇEKME
# ─────────────────────────────────────
@st.cache_data(ttl=3600)
def veri_cek_ve_hesapla():
    bitis = datetime.today().strftime('%Y-%m-%d')
    baslangic = (datetime.today() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
    satirlar = []
    for hisse in BIST50_HISSELER:
        try:
            time.sleep(0.5)
            df_h = yf.download(hisse, start=baslangic, end=bitis,
                               auto_adjust=True, progress=False)
            if df_h.empty:
                continue
            df_h.columns = [c[0] if isinstance(c, tuple) else c for c in df_h.columns]
            df_h['ticker'] = hisse
            df_h['tarih'] = df_h.index
            df_h = df_h.dropna(subset=['Close', 'Volume'])
            df_h = df_h.rename(columns={
                'Open': 'acilis', 'High': 'yuksek',
                'Low': 'dusuk', 'Close': 'kapanis', 'Volume': 'hacim'
            })
            satirlar.append(df_h)
        except:
            continue
    df = pd.concat(satirlar, ignore_index=True)
    df = df.sort_values(['ticker', 'tarih']).reset_index(drop=True)
    return teknik_hesapla(df)


# ─────────────────────────────────────
# SAYFA YAPISI
# ─────────────────────────────────────
st.set_page_config(page_title="BIST 50 Radar", page_icon="📡", layout="wide")

with st.sidebar:
    st.title("📡 BIST 50 Radar")
    st.markdown("*Teknik Analiz Sistemi*")
    st.divider()
    sayfa = st.radio("Sayfa Seç", [
        "🏠 Sabah Raporu",
        "🔍 Hisse Detay",
        "📚 El Kitabı"
    ])
    st.divider()
    st.caption("BIST 50 | 48 Hisse")
    st.caption("RSI · MACD · BB · MA · OBV")

# ─────────────────────────────────────
# SABAH RAPORU
# ─────────────────────────────────────
if sayfa == "🏠 Sabah Raporu":
    st.title("🏠 Sabah Raporu")

    with st.spinner("📡 Canlı veri çekiliyor... (~3 dakika)"):
        df = veri_cek_ve_hesapla()

    son_tarih = df['tarih'].max()
    bugun = df[df['tarih'] == son_tarih].copy()

    bugun['cakisma'] = bugun.apply(lambda row:
        '⚠️ Çelişkili'
        if ('DIV20' in str(row['aktif_sinyaller']) and
            row['Div_20g'] == '📉 Bearish' and
            row['MA_sinyal'] == 'Güçlü Yükseliş Trendi')
        else '✅ Tutarlı', axis=1)

    st.markdown(f"**Son işlem günü:** {son_tarih.strftime('%d %B %Y')}")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Toplam Hisse", len(bugun))
    c2.metric("🔴 Çok Güçlü", len(bugun[bugun['sinyal_seviyesi'] == '🔴 Çok Güçlü']))
    c3.metric("🟠 Güçlü", len(bugun[bugun['sinyal_seviyesi'] == '🟠 Güçlü']))
    c4.metric("⚠️ Çelişkili", len(bugun[bugun['cakisma'] == '⚠️ Çelişkili']))
    c5.metric("✅ Normal", len(bugun[bugun['sinyal_seviyesi'] == '✅ Normal']))

    st.divider()

    guclu = bugun[
        bugun['sinyal_seviyesi'].isin(['🔴 Çok Güçlü', '🟠 Güçlü'])
    ].sort_values('guc_skoru', ascending=False)

    if len(guclu) > 0:
        st.subheader(f"💪 Güçlü Sinyaller ({len(guclu)} hisse)")
        for _, row in guclu.iterrows():
            cakisma_icon = '⚠️' if row['cakisma'] == '⚠️ Çelişkili' else ''
            with st.expander(
                f"{row['sinyal_seviyesi']} **{row['ticker']}** "
                f"— {row['genel_yon']} — Skor: {row['guc_skoru']} {cakisma_icon}"
            ):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Kapanış", f"{row['kapanis']:.2f} TL")
                c2.metric("Fiyat %", f"{row['fiyat_degisim']:+.2f}%")
                c3.metric("RSI", f"{row['RSI']:.1f}")
                c4.metric("Hacim", f"{row['hacim_oran']:.2f}x")
                c5.metric("Güç Skoru", f"{row['guc_skoru']}/10")

                st.markdown(f"""
| Gösterge | Sinyal |
|----------|--------|
| MA Trend | {row['MA_sinyal']} |
| MACD | {row['MACD_sinyal']} |
| BB | {row['BB_sinyal']} |
| RSI | {row['RSI_sinyal']} |
| OBV | {row['OBV_sinyal']} |
| Div 10g | {row['Div_10g']} |
| Div 20g | {row['Div_20g']} |
| Cross | {row['Cross_sinyal']} |
                """)

                if row['cakisma'] == '⚠️ Çelişkili':
                    st.warning("⚠️ Trend güçlü görünüyor ama Bearish Divergence momentum zayıflıyor diyor. Dikkatli ol!")
    else:
        st.success("✅ Bugün güçlü sinyal yok — piyasa sakin.")

    st.divider()

    cross = bugun[bugun['Cross_sinyal'] != 'Yok']
    if len(cross) > 0:
        st.subheader("🌟 Cross Sinyalleri")
        for _, row in cross.iterrows():
            if row['Cross_sinyal'] == '🌟 Golden Cross':
                st.success(f"**{row['ticker']}** — {row['Cross_sinyal']} | MA50: {row['MA50']:.2f} | MA200: {row['MA200']:.2f}")
            else:
                st.error(f"**{row['ticker']}** — {row['Cross_sinyal']} | MA50: {row['MA50']:.2f} | MA200: {row['MA200']:.2f}")

    st.divider()
    st.subheader("📋 Tüm Hisseler")
    tablo = bugun[[
        'ticker', 'kapanis', 'fiyat_degisim',
        'RSI', 'sinyal_seviyesi', 'genel_yon',
        'guc_skoru', 'aktif_sinyaller'
    ]].sort_values('guc_skoru', ascending=False).reset_index(drop=True)
    tablo.columns = ['Hisse', 'Kapanış', 'Fiyat %', 'RSI',
                     'Sinyal', 'Yön', 'Skor', 'Aktif Göstergeler']
    tablo['Fiyat %'] = tablo['Fiyat %'].round(2)
    st.dataframe(tablo, use_container_width=True)

# ─────────────────────────────────────
# HİSSE DETAY
# ─────────────────────────────────────
elif sayfa == "🔍 Hisse Detay":
    st.title("🔍 Hisse Detay Analizi")

    with st.spinner("📡 Canlı veri çekiliyor..."):
        df = veri_cek_ve_hesapla()

    secili = st.selectbox("Hisse Seç", sorted(df['ticker'].unique()))
    d = df[df['ticker'] == secili].sort_values('tarih').copy()
    son = d.iloc[-1]

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Kapanış", f"{son['kapanis']:.2f} TL", f"{son['fiyat_degisim']:+.2f}%")
    c2.metric("RSI", f"{son['RSI']:.1f}")
    c3.metric("MACD", f"{son['MACD']:.3f}")
    c4.metric("BB Pozisyon", f"{son['BB_pozisyon']:.2f}")
    c5.metric("Güç Skoru", f"{son['guc_skoru']}/10")

    st.divider()
    st.subheader("📈 Fiyat Grafiği")
    gun_filtre = st.slider("Kaç gün göster?", 30, 750, 180)
    d_filtre = d.tail(gun_filtre)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Fiyat + MA + BB', 'MACD', 'RSI'),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05
    )

    fig.add_trace(go.Candlestick(
        x=d_filtre['tarih'],
        open=d_filtre['acilis'], high=d_filtre['yuksek'],
        low=d_filtre['dusuk'], close=d_filtre['kapanis'],
        name='Fiyat',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    for ma, renk in [('MA20', '#FFA500'), ('MA50', '#00BFFF'), ('MA200', '#FF69B4')]:
        fig.add_trace(go.Scatter(
            x=d_filtre['tarih'], y=d_filtre[ma],
            name=ma, line=dict(color=renk, width=1.5)
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d_filtre['tarih'], y=d_filtre['BB_upper'],
        line=dict(color='rgba(150,150,150,0.4)', dash='dot'),
        showlegend=False, name='BB Üst'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d_filtre['tarih'], y=d_filtre['BB_lower'],
        line=dict(color='rgba(150,150,150,0.4)', dash='dot'),
        fill='tonexty', fillcolor='rgba(150,150,150,0.05)',
        showlegend=False, name='BB Alt'
    ), row=1, col=1)

    renkler = ['#26a69a' if h >= 0 else '#ef5350' for h in d_filtre['MACD_histogram']]
    fig.add_trace(go.Bar(
        x=d_filtre['tarih'], y=d_filtre['MACD_histogram'],
        name='Histogram', marker_color=renkler
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d_filtre['tarih'], y=d_filtre['MACD'],
        name='MACD', line=dict(color='#00BFFF', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=d_filtre['tarih'], y=d_filtre['MACD_sinyal_cizgi'],
        name='Sinyal', line=dict(color='#FFA500', width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=d_filtre['tarih'], y=d_filtre['RSI'],
        name='RSI', line=dict(color='#9C27B0', width=1.5)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='red', opacity=0.05, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='green', opacity=0.05, row=3, col=1)

    fig.update_layout(
        height=700,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🎯 Güncel Sinyal Tablosu")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
| Gösterge | Değer | Sinyal |
|----------|-------|--------|
| RSI | {son['RSI']:.1f} | {son['RSI_sinyal']} |
| MA Trend | — | {son['MA_sinyal']} |
| MACD | {son['MACD']:.3f} | {son['MACD_sinyal']} |
| BB Pozisyon | {son['BB_pozisyon']:.2f} | {son['BB_sinyal']} |
| OBV | — | {son['OBV_sinyal']} |
| Div 10g | — | {son['Div_10g']} |
| Div 20g | — | {son['Div_20g']} |
| Cross | — | {son['Cross_sinyal']} |
        """)
    with col2:
        st.metric("Genel Yön", son['genel_yon'])
        st.metric("Güç Skoru", f"{son['guc_skoru']}/10")
        st.metric("Sinyal Seviyesi", son['sinyal_seviyesi'])
        st.metric("Aktif Göstergeler", son['aktif_sinyaller'])

# ─────────────────────────────────────
# EL KİTABI
# ─────────────────────────────────────
elif sayfa == "📚 El Kitabı":
    st.title("📚 Teknik Analiz El Kitabı")
    st.markdown("*BIST yatırımcısı için pratik referans kılavuzu*")

    bolum = st.selectbox("📖 Bölüm Seç", [
        "🗺️ Genel Bakış — Göstergeler Ne İşe Yarar?",
        "📈 Trend — Hareketli Ortalama (MA)",
        "⚡ Momentum — RSI",
        "🌊 İvme — MACD",
        "📊 Volatilite — Bollinger Bands",
        "💰 Hacim — OBV",
        "🔄 Divergence — En Güçlü Sinyal",
        "🎯 Güç Skoru Sistemi",
        "⚠️ Sınırlamalar & Riskler"
    ])

    # ─── GENEL BAKIŞ ───────────────────────
    if bolum == "🗺️ Genel Bakış — Göstergeler Ne İşe Yarar?":
        st.header("🗺️ Genel Bakış")

        st.info("""
        **Teknik analiz ne varsayar?**

        "Geçmiş fiyat ve hacim verisi gelecek hakkında bilgi içerir."
        Piyasada insan psikolojisi tekrar eder → pattern'ler tekrar eder.
        """)

        st.subheader("5 Gösterge — 5 Farklı Soru")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            | Gösterge | Sorusu |
            |----------|--------|
            | **MA** | Genel trend nereye gidiyor? |
            | **RSI** | Ne kadar aşırıya gitti? |
            | **MACD** | İvme artıyor mu, azalıyor mu? |
            | **BB** | Normal aralık dışına çıktı mı? |
            | **OBV** | Gerçek para nereye akıyor? |
            """)

        with col2:
            st.markdown("""
            **Önemli:** Tek gösterge yanıltıcı olabilir.

            Birden fazla gösterge aynı yönü gösteriyorsa
            → **Confluence (Yakınsama)** → güçlü sinyal!
```
            RSI < 30  AND
            BB Alt Kırılım AND
            MACD dönüyor AND
            OBV yükseliyor
            → 4 gösterge uyuşuyor → çok güçlü!
```
            """)

        st.divider()
        st.subheader("Güç Skoru Sistemi")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("✅ Normal", "0 puan", "Sinyal yok")
        col2.metric("🔵 Zayıf", "1 puan", "Tek gösterge")
        col3.metric("🟡 Orta", "2-3 puan", "Birkaç uyuşuyor")
        col4.metric("🟠 Güçlü", "4-5 puan", "Çoğu uyuşuyor")
        col5.metric("🔴 Çok Güçlü", "6+ puan", "Confluence!")

    # ─── MA ────────────────────────────────
    elif bolum == "📈 Trend — Hareketli Ortalama (MA)":
        st.header("📈 Hareketli Ortalama (Moving Average)")

        st.info("**Ne soruyor?** Genel trend nereye gidiyor?")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Türler")
            st.markdown("""
            **MA20** → 20 günlük ortalama
            Kısa vadeli trend. Swing trader kullanır.

            **MA50** → 50 günlük ortalama
            Orta vadeli trend. En yaygın kullanılan.

            **MA200** → 200 günlük ortalama
            Uzun vadeli trend. Kurumsal yatırımcı referansı.
            """)

        with col2:
            st.subheader("Sinyal Tablosu")
            st.markdown("""
            | Durum | Sinyal |
            |-------|--------|
            | Fiyat > MA50 > MA200 | 🟢 Güçlü Yükseliş |
            | Fiyat > MA50, MA50 < MA200 | 🟡 Toparlanma |
            | Fiyat < MA50, MA50 > MA200 | 🟡 Zayıflama |
            | Fiyat < MA50 < MA200 | 🔴 Güçlü Düşüş |
            """)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **🌟 Golden Cross — AL Sinyali**

            MA50, MA200'ü yukarı kesti

            → Kısa vade uzun vadeyi geçti
            → Yükseliş trendi başlıyor
            → Nadir ama güçlü sinyal (2 puan)

            Örnek: THYAO 2024 başı → Golden Cross
            → 6 ayda +%45 yükseldi
            """)

        with col2:
            st.error("""
            **💀 Death Cross — SAT Sinyali**

            MA50, MA200'ü aşağı kesti

            → Kısa vade uzun vadenin altına düştü
            → Düşüş trendi başlıyor
            → Nadir ama güçlü sinyal (2 puan)

            Örnek: AKBNK Eylül 2024 → Death Cross
            → 2 ayda -%18 düştü
            """)

        st.warning("""
        ⚠️ **Kısıtlama:** MA gecikmeli göstergedir.
        Trend başladıktan sonra sinyal verir — geç kalabilir.
        Haber bazlı ani hareketlerde hiç çalışmaz.
        """)

    # ─── RSI ───────────────────────────────
    elif bolum == "⚡ Momentum — RSI":
        st.header("⚡ RSI — Relative Strength Index")

        st.info("**Ne soruyor?** Hisse ne kadar aşırıya gitti?")

        st.subheader("Formül")
        st.code("""
RSI = 100 - (100 / (1 + RS))

RS = Son 14 günün ortalama yükselişi
     ÷
     Son 14 günün ortalama düşüşü
        """)

        st.subheader("Bölgeler")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.error("""
            **RSI > 80**
            Çok Aşırı Alınmış

            → Çok hızlı yükseldi
            → Sert düşüş riski yüksek
            → Yeni alım çok riskli
            """)
        with col2:
            st.warning("""
            **RSI 70-80**
            Aşırı Alınmış

            → Yükseliş güçlü
            → Geri çekilme gelebilir
            → Mevcut pozisyonu koru
            """)
        with col3:
            st.info("""
            **RSI 30-70**
            Normal Bölge

            → Trend devam ediyor
            → Diğer göstergelere bak
            → En sağlıklı bölge
            """)
        with col4:
            st.success("""
            **RSI < 30**
            Aşırı Satılmış

            → Çok hızlı düştü
            → Toparlanma olabilir
            → Alım fırsatı araştır
            """)

        st.divider()
        st.subheader("BIST Örneği")
        st.markdown("""
        **AKBNK Mart 2025 (İBB Krizi):**
```
        RSI = 18 → Çok Aşırı Satılmış
        → Piyasa aşırı tepki verdi
        → 2 hafta sonra +%12 toparlandı
        → RSI bu toparlanmayı önceden işaret etti!
```
        """)

        st.warning("""
        ⚠️ **Kısıtlama:** Güçlü trendde RSI uzun süre
        aşırı bölgede kalabilir!

        "RSI 80'de" → her zaman "düşecek" demek değil.
        Trend çok güçlüyse haftalarca 80 üstünde kalabilir.
        Mutlaka diğer göstergelerle teyit et!
        """)

    # ─── MACD ──────────────────────────────
    elif bolum == "🌊 İvme — MACD":
        st.header("🌊 MACD — Momentum & İvme")

        st.info("**Ne soruyor?** Momentum artıyor mu, azalıyor mu?")

        st.subheader("Formül")
        st.code("""
MACD Çizgisi   = EMA(12) - EMA(26)
Sinyal Çizgisi = EMA(9) × MACD
Histogram      = MACD - Sinyal Çizgisi
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sinyal Tablosu")
            st.markdown("""
            | MACD | Histogram | Sinyal |
            |------|-----------|--------|
            | > 0 | > 0 | 🟢 Güçlü Yükseliş |
            | > 0 | < 0 | 🟡 Yükseliş Zayıflıyor |
            | < 0 | > 0 | 🟡 Düşüş Zayıflıyor |
            | < 0 | < 0 | 🔴 Güçlü Düşüş |
            """)

        with col2:
            st.subheader("Araba Analojisi")
            st.markdown("""
```
            MACD(+) Histogram(+):
            Hızlanıyorsun → gaz basıyorsun 🚀

            MACD(+) Histogram(-):
            Hala hızlısın ama yavaşlıyorsun ⚠️

            MACD(-) Histogram(+):
            Yavaşlıyorsun ama frenden çekiyorsun 🔄

            MACD(-) Histogram(-):
            Fren basıyorsun → düşüş sürüyor 📉
```
            """)

        st.warning("""
        ⚠️ **Kısıtlama:** MACD gecikmeli göstergedir.
        Ani haber bazlı hareketlerde sinyal çok geç gelir.
        """)

    # ─── BOLLINGER ─────────────────────────
    elif bolum == "📊 Volatilite — Bollinger Bands":
        st.header("📊 Bollinger Bands — Volatilite")

        st.info("**Ne soruyor?** Fiyat normal aralığının dışına çıktı mı?")

        st.subheader("Formül")
        st.code("""
Orta Bant = MA(20)
Üst Bant  = MA(20) + 2 × Standart Sapma
Alt Bant  = MA(20) - 2 × Standart Sapma

BB Pozisyon = (Fiyat - Alt Bant) / (Üst Bant - Alt Bant)
→ 0 = alt bantta
→ 0.5 = tam ortada
→ 1 = üst bantta
→ > 1 = üst kırılım!
→ < 0 = alt kırılım!

İstatistiksel anlam:
%95 ihtimalle fiyat bant içinde kalır
Bant dışı = nadir olay = potansiyel anomali!
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.error("""
            **Üst Kırılım (BB > 1)**

            → Fiyat normalin çok üstünde
            → İki senaryo:
              1. Güçlü momentum → devam eder
              2. Aşırı reaksiyon → geri çekilir

            RSI ile teyit et!
            RSI > 70 ise → geri çekilme riski yüksek
            RSI < 70 ise → momentum devam edebilir
            """)

        with col2:
            st.success("""
            **Alt Kırılım (BB < 0)**

            → Fiyat normalin çok altında
            → İki senaryo:
              1. Düşüş devam eder
              2. Aşırı reaksiyon → toparlanır

            RSI ile teyit et!
            RSI < 30 ise → toparlanma ihtimali yüksek
            RSI > 30 ise → düşüş devam edebilir
            """)

        st.divider()
        st.subheader("🔥 BB Squeeze")
        st.warning("""
        **Bantlar çok daraldı → Büyük hareket geliyor!**

        Piyasa bir şey bekliyor → yatırımcılar kararsız → az işlem
        → Volatilite düşüyor → bantlar daralıyor
        → Haber gelince PATLAMA!
        → Hangi yöne? Bilinmez — ama büyük hareket geliyor.

        Kural: BB Squeeze + diğer göstergeler yönü söyler.
        """)

    # ─── OBV ───────────────────────────────
    elif bolum == "💰 Hacim — OBV":
        st.header("💰 OBV — On Balance Volume")

        st.info("**Ne soruyor?** Gerçek para nereye akıyor?")

        st.subheader("Formül")
        st.code("""
Yükseliş günü: OBV = OBV(dün) + Hacim
Düşüş günü:   OBV = OBV(dün) - Hacim

Mantık: Yükselen günlerde hacim fazlaysa
        → Para hisseye giriyor
        → Büyük yatırımcı alıyor
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **OBV > OBV_MA20**
            = Alım Baskısı

            → Yükselen günlerde hacim fazla
            → Para hisseye giriyor
            → Büyük yatırımcı topluyor
            → Yükseliş sürebilir
            """)

        with col2:
            st.error("""
            **OBV < OBV_MA20**
            = Satış Baskısı

            → Düşen günlerde hacim fazla
            → Para hisseden çıkıyor
            → Büyük yatırımcı satıyor
            → Düşüş sürebilir
            """)

        st.divider()
        st.subheader("OBV Divergence — En Güçlü Hacim Sinyali")
        st.markdown("""
```
        Fiyat DÜŞÜYOR ama OBV YÜKSELIYOR:
        → Büyük yatırımcı düşüşte topluyor
        → "Sessiz alım" var
        → Yakında fiyat da yükselir

        Fiyat YÜKSELIYOR ama OBV DÜŞÜYOR:
        → Büyük yatırımcı yükselişte satıyor
        → "Dağıtım" yapıyor
        → Zayıf yükseliş — yakında düşer
```
        """)

    # ─── DIVERGENCE ────────────────────────
    elif bolum == "🔄 Divergence — En Güçlü Sinyal":
        st.header("🔄 Divergence — Fiyat ve Momentum Uyumsuzluğu")

        st.info("""
        **Ne soruyor?** Fiyat ve RSI aynı yönde mi?

        Teknik analizin en güçlü sinyali olarak kabul edilir.
        Fiyat ile RSI'ın ters yönde hareket etmesi.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **📈 Bullish Divergence**
            (Pozitif Uyumsuzluk)
```
            Fiyat:  düşük → DAHA DÜŞÜK  ↘
            RSI:    düşük → DAHA YÜKSEK ↗
```

            Anlam:
            → Fiyat düşüyor ama momentum toparlanıyor
            → Satış baskısı azalıyor
            → Yakında fiyat da toparlanır

            Bizim kriterlerimiz:
            → Fiyat -%3'ten fazla düşmeli
            → RSI +5 puandan fazla yükselmeli
            → RSI < 50 olmalı (zayıf bölgede)
            """)

        with col2:
            st.error("""
            **📉 Bearish Divergence**
            (Negatif Uyumsuzluk)
```
            Fiyat:  yüksek → DAHA YÜKSEK ↗
            RSI:    yüksek → DAHA DÜŞÜK  ↘
```

            Anlam:
            → Fiyat yükseliyor ama momentum zayıflıyor
            → Alım baskısı azalıyor
            → Yakında fiyat da düşer

            Bizim kriterlerimiz:
            → Fiyat +%3'ten fazla yükselmeli
            → RSI -5 puandan fazla düşmeli
            → RSI > 50 olmalı (güçlü bölgede)
            """)

        st.divider()
        st.subheader("Pencere Önemi")
        st.markdown("""
        | Pencere | Güvenilirlik | Puan |
        |---------|-------------|------|
        | 10 gün | 🟡 Orta | 1 puan |
        | 20 gün | ✅ Yüksek | 2 puan |

        **20 günlük divergence daha güvenilir** — daha uzun süreçte oluştu.
        Bu yüzden güç skorunda 2 puan veriyor.
        """)

        st.warning("""
        ⚠️ **Önemli:** Divergence tek başına yeterli değil!

        Fiyat düşmeye devam edebilir. Divergence sadece
        "momentum zayıflıyor" diyor — "kesin toparlanacak" demez.
        Mutlaka RSI seviyesi ve diğer göstergelerle birlikte değerlendir.
        """)

    # ─── GÜÇ SKORU ─────────────────────────
    elif bolum == "🎯 Güç Skoru Sistemi":
        st.header("🎯 Güç Skoru Sistemi")

        st.info("""
        Her gösterge bağımsız sinyal verir.
        Güç skoru bu sinyallerin toplamı.
        Yüksek skor = daha fazla gösterge aynı yönde = daha güvenilir.
        """)

        st.subheader("Puan Tablosu")
        st.markdown("""
        | Gösterge | Koşul | Puan | Yön |
        |----------|-------|------|-----|
        | RSI | < 30 veya > 70 | 1 | ↑ veya ↓ |
        | MA | Güçlü trend | 1 | ↑ veya ↓ |
        | MACD | Güçlü sinyal | 1 | ↑ veya ↓ |
        | BB | Kırılım | 1 | ↑ veya ↓ |
        | OBV | Baskı | 1 | ↑ veya ↓ |
        | Div 10g | Bullish/Bearish | 1 | ↑ veya ↓ |
        | Div 20g | Bullish/Bearish | **2** | ↑ veya ↓ |
        | Cross | Golden/Death | **2** | ↑ veya ↓ |
        """)

        st.subheader("Seviye Tablosu")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.error("🔴 Çok Güçlü\n\n6+ puan\n\nConfluence!")
        col2.warning("🟠 Güçlü\n\n4-5 puan\n\nGüçlü sinyal")
        col3.info("🟡 Orta\n\n2-3 puan\n\nOrta sinyal")
        col4.markdown("🔵 Zayıf\n\n1 puan\n\nTek gösterge")
        col5.success("✅ Normal\n\n0 puan\n\nSinyal yok")

        st.divider()
        st.subheader("⚠️ Çelişki Kontrolü")
        st.warning("""
        Yüksek güç skoru tek başına yeterli değil!

        Örnek: ASTOR bugün güç skoru 5
        → MA + MACD + OBV: Yükseliş
        → DIV20: Bearish (düşüş uyarısı!)

        → Çelişkili sinyal → dikkatli ol!
        → Trend güçlü ama momentum zayıflıyor
        → Bu sistem çelişkileri otomatik flagliyor
        """)

    # ─── SINIRLAMALAR ──────────────────────
    elif bolum == "⚠️ Sınırlamalar & Riskler":
        st.header("⚠️ Sınırlamalar & Riskler")

        st.error("""
        **Bu sistem ne DEĞİLDİR:**

        ❌ Al/sat robotu değil
        ❌ Kesin tahmin aracı değil
        ❌ Yatırım tavsiyesi değil
        ❌ Geçmiş performans gelecek garantisi değildir
        """)

        st.subheader("Ne Zaman Çalışmaz?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **❌ Haber bazlı ani hareketler:**
```
            "CEO tutuklandı" → -%20 tek günde
            Hiçbir gösterge bunu bilemez
            RSI, BB, MACD → hepsi geç kalır
```

            **❌ Manipülasyon:**
```
            Bear raid, pump & dump
            Pattern'ler yapay → göstergeler yanıltır
            GUBRF Aralık 2023 bunu gösterdi
```
            """)

        with col2:
            st.markdown("""
            **❌ Kriz dönemleri:**
```
            Seçim, siyasi kriz, global çöküş
            Tüm korelasyonlar bozulur
            "Bu sefer farklı" sendromu
```

            **❌ Az işlem gören hisseler:**
```
            Düşük hacim → göstergeler güvenilmez
            OBV özellikle yanıltıcı olur
            BIST 50 hisseleri için daha güvenilir
```
            """)

        st.divider()
        st.subheader("✅ Bu Sistem Ne İçin Kullanılabilir?")
        st.success("""
        **Filtreleme aracı:**
        48 hisseyi takip etmek zor
        → Sistem güçlü sinyalleri öne çıkarır
        → Sen sadece bu hisseleri araştırırsın

        **Risk uyarısı:**
        Portföyündeki hisse sinyal verdi
        → "Bir şey mi oldu?" diye araştır
        → Stop-loss seviyeni gözden geçir

        **Araştırma başlangıcı:**
        Teknik sinyal → haberleri araştır → karar ver
        Teknik analiz tek başına yeterli değil!
        """)
