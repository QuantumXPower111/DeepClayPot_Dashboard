"""
Deep Claypot Dashboard Financial, Inc. (EODHD Edition)
Sentiment-Aware Trading Dashboard with Portfolio Allocation
Author: GLM 4.7 Reasoning
Date: February 2026
Production version fully powered by EODHD APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import os

# =============================================================================
# Configuration & Constants
# =============================================================================

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class Settings:
    """Application Settings."""
    # Updated with provided API Token
    eodhd_api_token: str = "6999b5632169a2.16580729" 
    page_title: str = "Deep Claypot Dashboard (EODHD Edition)"
    page_icon: str = "📈"
    cache_ttl: int = 300  # 5 minutes
    
    @classmethod
    def load_env(cls) -> 'Settings':
        # Checks environment variable first, otherwise uses the default above
        return cls(
            eodhd_api_token=os.getenv("EODHD_API_TOKEN", "6999b5632169a2.16580729")
        )

settings = Settings.load_env()

# Strategic Portfolio Allocation (Q2 2026)
# Format: Ticker with .US extension for EODHD
PORTFOLIO_ALLOCATION: Dict[str, float] = {
    'NVDA.US': 0.20,
    'LMT.US': 0.20,
    'ASML.US': 0.30,
    'CCO.US': 0.10,
    'GLD.US': 0.10,
    'MU.US': 0.10,
}

COMPANY_NAMES: Dict[str, str] = {
    'NVDA.US': 'NVIDIA Corporation',
    'LMT.US': 'Lockheed Martin Corp',
    'ASML.US': 'ASML Holding NV',
    'CCO.US': 'Clear Channel Outdoor',
    'GLD.US': 'SPDR Gold Trust',
    'MU.US': 'Micron Technology Inc'
}

SECTOR_CLASSIFICATION: Dict[str, str] = {
    'NVDA.US': 'Technology',
    'LMT.US': 'Aerospace & Defense',
    'ASML.US': 'Semiconductors',
    'CCO.US': 'Media & Advertising',
    'GLD.US': 'Commodities (Gold)',
    'MU.US': 'Semiconductors'
}

CHART_COLORS: List[str] = ['#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE']
TIME_PERIODS: List[str] = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
PORTFOLIO_SCALE_FACTOR: int = 10000

# =============================================================================
# Data Fetching Module (EODHD Primary)
# =============================================================================

@dataclass
class StockData:
    ticker: str
    company_name: str
    sector: str
    current_price: Optional[float] = None
    change: float = 0.0
    change_pct: float = 0.0
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume: int = 0
    market_cap: int = 0
    pe_ratio: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.current_price is not None and not self.history.empty

class EODHDDataFetcher:
    """Handles data retrieval using EODHD APIs."""
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_token: str):
        self.api_token = api_token
    
    def _get_period_dates(self, period: str) -> Tuple[str, str]:
        """Calculate date range from period string."""
        end_date = datetime.now()
        start_date = end_date
        
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
            
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    @st.cache_data(ttl=settings.cache_ttl)
    def get_historical_prices(_self, ticker: str, period: str = "1mo") -> Optional[StockData]:
        """
        Fetch historical prices from EODHD Historical Data API.
        Implements logic from insightbig.com tutorial.
        """
        try:
            start_date, end_date = _self._get_period_dates(period)
            
            # EODHD API Endpoint for Historical Prices
            url = f"{_self.BASE_URL}/eod/{ticker}"
            params = {
                'api_token': _self.api_token,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Standardize Column Names
            df.columns = [c.lower() for c in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Calculate Metrics
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
            
            # Calculate 52-week high/low from the fetched data (assuming sufficient history)
            week_52_high = df['high'].max()
            week_52_low = df['low'].min()
            
            return StockData(
                ticker=ticker,
                company_name=COMPANY_NAMES.get(ticker, ticker),
                sector=SECTOR_CLASSIFICATION.get(ticker, 'Other'),
                current_price=current_price,
                change=change,
                change_pct=change_pct,
                history=df,
                volume=int(df['volume'].iloc[-1]),
                week_52_high=week_52_high,
                week_52_low=week_52_low
            )
            
        except Exception as e:
            logger.error(f"EODHD Fetch Error for {ticker}: {e}")
            return None

    @st.cache_data(ttl=600) # Cache fundamentals longer
    def get_fundamental_data(_self, ticker: str) -> Dict[str, Any]:
        """
        Fetch additional data (Market Cap, etc.) using Screener API logic.
        Based on python.plainenglish.io tutorial.
        """
        try:
            # We use the screener to get specific static data like Market Cap
            filters = f'[["code","=","{ticker}"]]'
            url = f"{_self.BASE_URL}/screener"
            params = {
                'api_token': _self.api_token,
                'filters': filters,
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get('data'):
                    return json_data['data'][0]
        except Exception as e:
            logger.warning(f"Could not fetch fundamental data for {ticker}: {e}")
        return {}

    @st.cache_data(ttl=settings.cache_ttl)
    def get_news_headlines(_self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news headlines using EODHD News API."""
        try:
            url = f"{_self.BASE_URL}/news"
            params = {
                's': ticker, 
                'api_token': _self.api_token,
                'limit': 10,
                'offset': 0,
                'fmt': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")
            return []

# =============================================================================
# Fallback / Helper Modules
# =============================================================================

def get_yfinance_simple_info(ticker: str) -> Dict[str, Any]:
    """
    Simple yfinance fallback for specific metrics not easily calculated 
    from raw price history (e.g., Trailing P/E).
    Used strictly as a supplementary data source.
    """
    try:
        import yfinance as yf
        # Clean ticker for yfinance
        clean_t = ticker.replace(".US", "")
        stock = yf.Ticker(clean_t)
        info = stock.info
        return {
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except Exception:
        return {'pe_ratio': None, 'dividend_yield': 0.0}

# =============================================================================
# Sentiment Analysis Module
# =============================================================================

@dataclass
class SentimentResult:
    headline: str
    source: str
    date: str
    compound: float
    label: str

class SentimentAnalyzer:
    _instance = None
    _analyzer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        self._analyzer = SentimentIntensityAnalyzer()
    
    @st.cache_data(ttl=settings.cache_ttl)
    def analyze(_self, articles: List[Dict]) -> List[SentimentResult]:
        results = []
        for article in articles:
            title = article.get('title', '')
            scores = _self._analyzer.polarity_scores(title)
            compound = scores['compound']
            label = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
            
            results.append(SentimentResult(
                headline=title,
                source=article.get('source', 'EODHD'),
                date=article.get('date', ''),
                compound=compound,
                label=label
            ))
        return results

# =============================================================================
# Charting Module
# =============================================================================

class ChartBuilder:
    
    def create_pie_chart(self):
        fig = go.Figure(data=[go.Pie(
            labels=[f"{k} ({COMPANY_NAMES[k].split()[0]})" for k in PORTFOLIO_ALLOCATION.keys()],
            values=[v * 100 for v in PORTFOLIO_ALLOCATION.values()],
            hole=0.4,
            marker_colors=CHART_COLORS
        )])
        fig.update_layout(
            title="Strategic Asset Allocation",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        return fig
    
    def create_interactive_price_chart(self, stock_data: StockData, ticker: str):
        df = stock_data.history.copy()
        
        # Technical Indicators
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['Bollinger_Upper'] = df['MA20'] + (df['close'].rolling(window=20).std() * 2)
        df['Bollinger_Lower'] = df['MA20'] - (df['close'].rolling(window=20).std() * 2)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{ticker} Price Action & Technicals", "Volume", "Momentum / Volatility")
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="Price"
        ), row=1, col=1)
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name='MA50'), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], line=dict(color='gray', width=0, dash='dot'), name='BB Upper', fill='tonexty'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], line=dict(color='gray', width=0, dash='dot'), name='BB Lower'), row=1, col=1)
        
        # Volume
        colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name="Volume"), row=2, col=1)
        
        # Simple Volatility (Rolling Std Dev)
        df['Volatility'] = df['close'].pct_change().rolling(window=20).std()
        fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], fill='tozeroy', line=dict(color='purple'), name="20d Vol"), row=3, col=1)
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", row=3, col=1)
        
        return fig

    def create_sentiment_gauge(self, score: float):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Avg Sentiment Score"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.33], 'color': "lightcoral"},
                    {'range': [-0.33, 0.33], 'color': "lightgray"},
                    {'range': [0.33, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        return fig

# =============================================================================
# Main Application
# =============================================================================

def main():
    # Page Setup
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 800; margin-bottom: 0.5rem; }
        .metric-box { background-color: #F8FAFC; padding: 1rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .ticker-tab { font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize Fetchers
    fetcher = EODHDDataFetcher(settings.eodhd_api_token)
    analyzer = SentimentAnalyzer()
    chart_builder = ChartBuilder()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.info("**API Provider:** EODHD")
        st.caption("Powered by EODHD Historical & News APIs")
        
        period = st.selectbox("Time Horizon", TIME_PERIODS, index=3)
        
        st.markdown("---")
        st.markdown("### Strategic Allocation")
        alloc_df = pd.DataFrame([
            {'Ticker': k, '%': f"{v*100:.0f}%", 'Sector': SECTOR_CLASSIFICATION.get(k, 'N/A')} 
            for k, v in PORTFOLIO_ALLOCATION.items()
        ])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    # Header
    st.markdown('<p class="main-header">Deep Claypot Dashboard <span style="font-size:0.6em; color:gray;">(EODHD Edition)</span></p>', unsafe_allow_html=True)
    
    # Data Loading Section
    st.markdown("### Real-Time Market Data")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    portfolio_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(PORTFOLIO_ALLOCATION.keys()):
        status_text.text(f"Fetching EODHD data for {ticker}...")
        # Fetch Price Data
        stock = fetcher.get_historical_prices(ticker, period)
        
        if stock:
            # Enrich with Screener/Fundamental Data
            fundamentals = fetcher.get_fundamental_data(ticker)
            if fundamentals:
                stock.market_cap = fundamentals.get('market_capitalization', 0)
            
            # Fallback to yfinance for PE/Div if not in Screener result
            yf_data = get_yfinance_simple_info(ticker)
            if not stock.pe_ratio and yf_data.get('pe_ratio'):
                stock.pe_ratio = yf_data['pe_ratio']
            
            portfolio_data[ticker] = stock
        else:
            failed_tickers.append(ticker)
        
        progress_bar.progress((i + 1) / len(PORTFOLIO_ALLOCATION))
        
    status_text.empty()
    progress_bar.empty()
    
    if failed_tickers:
        st.warning(f"Data unavailable for: {', '.join(failed_tickers)}")
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_val = sum([d.current_price * PORTFOLIO_ALLOCATION[d.ticker] * PORTFOLIO_SCALE_FACTOR for d in portfolio_data.values()])
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Portfolio Value", f"${total_val:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Active Positions", len(portfolio_data))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        # Avg Change across portfolio
        avg_change = np.mean([d.change_pct for d in portfolio_data.values()]) if portfolio_data else 0
        st.metric("Avg Daily Change", f"{avg_change:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Data Source", "EODHD API")
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts Row 1: Allocation & Summary Table
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.plotly_chart(chart_builder.create_pie_chart(), use_container_width=True)
        
    with c2:
        st.markdown("#### Live Holdings Table")
        summary_data = []
        for ticker, data in portfolio_data.items():
            summary_data.append({
                'Ticker': ticker,
                'Sector': data.sector,
                'Price': f"${data.current_price:.2f}",
                'Change %': f"{data.change_pct:.2f}%",
                'Vol': f"{data.volume/1e6:.1f}M",
                '52w H/L': f"${data.week_52_low:.2f} - ${data.week_52_high:.2f}"
            })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    # Detailed Analysis Section
    st.markdown("---")
    st.markdown("## Deep Dive Analysis")
    
    # Tabs for each stock
    tabs = st.tabs(list(PORTFOLIO_ALLOCATION.keys()))
    
    for i, ticker in enumerate(PORTFOLIO_ALLOCATION.keys()):
        with tabs[i]:
            if ticker not in portfolio_data:
                st.error(f"Data unavailable for {ticker}")
                continue
                
            stock = portfolio_data[ticker]
            
            # Layout for individual stock
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.subheader(f"Price Action: {ticker}")
                st.plotly_chart(chart_builder.create_interactive_price_chart(stock, ticker), use_container_width=True)
            
            with col_b:
                st.subheader("Market Sentiment")
                
                # Fetch News
                with st.spinner(f"Scanning EODHD News for {ticker}..."):
                    news = fetcher.get_news_headlines(ticker)
                    sentiments = analyzer.analyze(news)
                
                if sentiments:
                    avg_score = np.mean([s.compound for s in sentiments])
                    st.plotly_chart(chart_builder.create_sentiment_gauge(avg_score), use_container_width=True)
                    
                    # Recent Headlines
                    st.markdown("#### Recent Headlines")
                    for s in sentiments[:5]:
                        color = "green" if s.label == 'Positive' else ("red" if s.label == 'Negative' else "gray")
                        st.markdown(f"<span style='color:{color}; font-size:0.9em;'>[{s.label}]</span> **{s.headline}**", unsafe_allow_html=True)
                        st.caption(f"{s.date} | {s.source}")
                        st.markdown("---")
                else:
                    st.info("No recent news data found via EODHD.")

    # Footer
    st.markdown("---")
    st.caption("Data provided by EODHD APIs. Market data delayed per exchange requirements. Sentiment analysis powered by NLTK VADER.")

if __name__ == "__main__":
    main()
