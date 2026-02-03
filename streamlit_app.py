"""
Streamlit Dashboard for Momentum Backtester.
Interactive web UI for running backtests and viewing results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict
import os

# Import backtester components
from polygon_fetcher import PolygonFetcher, DataManager
from backtester import Backtester, Portfolio
from context_engine import ContextEngine
from analyzer import TradeAnalyzer
from setups import ORBSetup, EpisodicPivotSetup, DelayedReactionSetup, EarningsPlaySetup
import config

# Page config
st.set_page_config(
    page_title="Momentum Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stMetric {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #e5e7eb !important;
    }
    .stSelectbox label, .stMultiSelect label, .stDateInput label {
        color: #9ca3af !important;
    }
</style>
""", unsafe_allow_html=True)

# Preset ticker lists
TICKER_PRESETS = {
    "-- Select --": [],  # Empty option
    "Mega Tech": ["NVDA", "AMD", "TSLA", "META", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "MU", "MRVL", "QCOM", "INTC", "ASML"],
    "Software/SaaS": ["CRM", "NOW", "ADBE", "SNOW", "PLTR", "NET", "DDOG", "ZS"],
    "Fintech": ["SQ", "COIN", "PYPL", "HOOD", "SOFI", "AFRM"],
    "Custom": []
}

# Catalyst types for auto-population
CATALYST_TYPES = {
    "None": None,
    "Earnings": "earnings",
    "Analyst Upgrade": "analyst_upgrade",
    "Analyst Downgrade": "analyst_downgrade",
    "Major News": "major_news"
}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_catalyst_tickers(catalyst_type: str, start_date: str, end_date: str, fetcher: PolygonFetcher = None) -> List[str]:
    """
    Fetch tickers with specific catalyst events in the date range.
    Uses Polygon.io API to find relevant stocks.
    """
    if fetcher is None:
        fetcher = PolygonFetcher()

    tickers = []

    try:
        if catalyst_type == "earnings":
            # Search for earnings announcements via ticker news
            # Use grouped daily to find high-volume gap days (proxy for earnings)
            for date_offset in range(min(30, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days)):
                check_date = (pd.to_datetime(start_date) + timedelta(days=date_offset)).strftime("%Y-%m-%d")
                data = fetcher.get_grouped_daily(check_date)
                if not data.empty and 'T' in data.columns:
                    # Look for stocks with >4% gap and high volume (earnings proxy)
                    if 'c' in data.columns and 'o' in data.columns and 'v' in data.columns:
                        data['gap_pct'] = ((data['o'] - data['c'].shift(1)) / data['c'].shift(1)) * 100
                        high_gap = data[abs(data.get('gap_pct', 0)) > 4]['T'].tolist()
                        tickers.extend(high_gap[:10])  # Limit per day
                if len(tickers) >= 20:
                    break

        elif catalyst_type in ["analyst_upgrade", "analyst_downgrade"]:
            # For analyst ratings, we use news search as proxy
            # Polygon free tier has limited news access, so use top movers
            data = fetcher.get_grouped_daily(start_date)
            if not data.empty and 'T' in data.columns and 'c' in data.columns and 'o' in data.columns:
                data['change_pct'] = ((data['c'] - data['o']) / data['o']) * 100
                if catalyst_type == "analyst_upgrade":
                    top_movers = data.nlargest(15, 'change_pct')['T'].tolist()
                else:
                    top_movers = data.nsmallest(15, 'change_pct')['T'].tolist()
                tickers.extend(top_movers)

        elif catalyst_type == "major_news":
            # High volume + high volatility days = major news proxy
            data = fetcher.get_grouped_daily(start_date)
            if not data.empty and 'T' in data.columns:
                if 'v' in data.columns and 'h' in data.columns and 'l' in data.columns and 'c' in data.columns:
                    data['range_pct'] = ((data['h'] - data['l']) / data['c']) * 100
                    high_vol = data[data['range_pct'] > 5].nlargest(20, 'v')['T'].tolist()
                    tickers.extend(high_vol)

    except Exception as e:
        st.warning(f"Catalyst search error: {e}")

    # Clean and dedupe tickers
    tickers = list(set([t for t in tickers if t and isinstance(t, str) and len(t) <= 5 and t.isalpha()]))
    return tickers[:20]  # Limit to 20 tickers


def generate_pooled_results(trades_df: pd.DataFrame) -> Dict:
    """
    Generate pooled/aggregated results across all tickers.
    Returns best setup recommendations based on aggregate performance.
    """
    if trades_df.empty:
        return {}

    results = {
        "total_tickers": trades_df['ticker'].nunique(),
        "total_trades": len(trades_df),
        "best_setups": [],
        "recommendations": []
    }

    # Analyze by setup type
    setup_stats = []
    for setup_type in trades_df['setup_type'].unique():
        subset = trades_df[trades_df['setup_type'] == setup_type]
        wins = subset[subset['pnl'] > 0]['pnl'].sum()
        losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
        pf = wins / losses if losses > 0 else float('inf')
        win_rate = (subset['pnl'] > 0).mean()

        setup_stats.append({
            'setup': setup_type,
            'trades': len(subset),
            'win_rate': win_rate,
            'profit_factor': min(pf, 10),
            'total_pnl': subset['pnl'].sum(),
            'avg_r': subset['r_multiple'].mean() if 'r_multiple' in subset.columns else 0
        })

    # Sort by profit factor
    setup_stats = sorted(setup_stats, key=lambda x: x['profit_factor'], reverse=True)
    results['best_setups'] = setup_stats[:3]

    # Generate recommendations
    if setup_stats:
        best = setup_stats[0]
        results['recommendations'].append(
            f"Based on {results['total_tickers']} stocks, **{best['setup']}** shows the best performance "
            f"with {best['win_rate']:.0%} win rate and {best['profit_factor']:.1f} profit factor."
        )

        # Context-based recommendations
        if 'ctx_market_regime' in trades_df.columns:
            regime_perf = trades_df.groupby('ctx_market_regime')['pnl'].sum()
            if not regime_perf.empty:
                best_regime = regime_perf.idxmax()
                results['recommendations'].append(
                    f"Best market regime for these stocks: **{best_regime}**"
                )

    return results


def create_equity_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create equity curve chart."""
    if trades_df.empty:
        return go.Figure()

    df = trades_df.sort_values('entry_time').copy()
    df['cumulative_pnl'] = df['pnl'].cumsum() + config.INITIAL_CAPITAL

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['entry_time'],
        y=df['cumulative_pnl'],
        mode='lines',
        name='Equity',
        line=dict(color='#22c55e', width=2),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.1)'
    ))

    # Add drawdown shading
    peak = df['cumulative_pnl'].cummax()
    drawdown = (df['cumulative_pnl'] - peak) / peak * 100

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )

    return fig


def create_setup_comparison(trades_df: pd.DataFrame) -> go.Figure:
    """Create setup comparison bar chart."""
    if trades_df.empty:
        return go.Figure()

    stats = []
    for setup in trades_df['setup_type'].unique():
        subset = trades_df[trades_df['setup_type'] == setup]
        wins = subset[subset['pnl'] > 0]['pnl'].sum()
        losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
        pf = wins / losses if losses > 0 else 5

        stats.append({
            'setup': setup,
            'trades': len(subset),
            'win_rate': (subset['pnl'] > 0).mean() * 100,
            'profit_factor': min(pf, 5),
            'total_pnl': subset['pnl'].sum()
        })

    df = pd.DataFrame(stats)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Win Rate by Setup', 'Profit Factor by Setup')
    )

    colors = px.colors.qualitative.Set2[:len(df)]

    fig.add_trace(
        go.Bar(x=df['setup'], y=df['win_rate'], marker_color=colors, name='Win Rate'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=df['setup'], y=df['profit_factor'], marker_color=colors, name='Profit Factor'),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False
    )

    return fig


def create_pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Create PnL distribution histogram."""
    if trades_df.empty:
        return go.Figure()

    fig = go.Figure()

    winners = trades_df[trades_df['pnl'] > 0]['pnl']
    losers = trades_df[trades_df['pnl'] <= 0]['pnl']

    fig.add_trace(go.Histogram(
        x=winners, name='Winners',
        marker_color='#22c55e', opacity=0.7
    ))

    fig.add_trace(go.Histogram(
        x=losers, name='Losers',
        marker_color='#ef4444', opacity=0.7
    ))

    fig.update_layout(
        title="PnL Distribution",
        xaxis_title="PnL ($)",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        barmode='overlay'
    )

    return fig


def create_context_heatmap(trades_df: pd.DataFrame) -> go.Figure:
    """Create context performance heatmap."""
    if trades_df.empty:
        return go.Figure()

    ctx_cols = [col for col in trades_df.columns
                if col.startswith('ctx_') and trades_df[col].dtype == 'object']

    if not ctx_cols:
        return go.Figure()

    # Build heatmap for top 3 context variables
    data = []
    y_labels = []
    x_labels = set()

    for col in ctx_cols[:3]:
        var_name = col.replace('ctx_', '')
        y_labels.append(var_name)

        row_data = {}
        for val in trades_df[col].dropna().unique():
            subset = trades_df[trades_df[col] == val]
            if len(subset) >= 3:
                wins = subset[subset['pnl'] > 0]['pnl'].sum()
                losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
                pf = wins / losses if losses > 0 else 3
                row_data[str(val)[:12]] = min(pf, 3)
                x_labels.add(str(val)[:12])

        data.append(row_data)

    x_labels = sorted(list(x_labels))
    z_data = [[d.get(x, np.nan) for x in x_labels] for d in data]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        showscale=True
    ))

    fig.update_layout(
        title="Context Performance Heatmap (Profit Factor)",
        template="plotly_dark",
        height=300
    )

    return fig


def run_backtest_with_progress(tickers, start_date, end_date, selected_setups):
    """Run backtest with progress indicator."""

    # Build setup list
    setup_map = {
        'ORB 3m': ORBSetup(timeframe=3),
        'ORB 5m': ORBSetup(timeframe=5),
        'ORB 15m': ORBSetup(timeframe=15),
        'ORB 60m': ORBSetup(timeframe=60),
        'Episodic Pivots': EpisodicPivotSetup(),
        'Delayed Reactions': DelayedReactionSetup(),
        'Earnings Plays': EarningsPlaySetup(play_type="post_momentum"),
    }

    setups = [setup_map[s] for s in selected_setups if s in setup_map]

    if not setups:
        st.error("Please select at least one setup")
        return None

    # Create backtester
    backtester = Backtester(setups=setups)

    # Run backtest
    progress_bar = st.progress(0, text="Initializing...")

    try:
        progress_bar.progress(10, text="Loading market data (parallel fetch)...")

        # Run the backtest
        trades_df = backtester.run(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            include_minute_data=False  # Disable minute data for faster runs
        )

        progress_bar.progress(100, text="Complete!")

        return trades_df, backtester.get_equity_curve()

    except Exception as e:
        import traceback
        st.error(f"Backtest error: {str(e)}")
        st.expander("Error details").code(traceback.format_exc())
        return None, None


def main():
    # Header
    st.title("📈 Momentum Backtester")
    st.markdown("*Analyze ORB, Episodic Pivots, Delayed Reactions & Earnings Plays*")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ticker Selection
        st.subheader("Tickers")
        preset = st.selectbox("Preset Watchlist", list(TICKER_PRESETS.keys()))

        tickers = []

        if preset == "Custom":
            ticker_input = st.text_area(
                "Enter tickers (comma separated)",
                "NVDA, AMD, TSLA, META"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        elif preset == "-- Select --":
            # Empty watchlist - show catalyst option
            st.info("Select a catalyst type below to auto-populate tickers")
        else:
            tickers = TICKER_PRESETS[preset]
            if tickers:
                st.write(f"Selected: {', '.join(tickers)}")

        # Catalyst Type Selection
        st.subheader("Catalyst Filter")
        catalyst = st.selectbox(
            "Auto-populate by catalyst",
            list(CATALYST_TYPES.keys()),
            help="Find stocks with specific catalyst events in your date range"
        )

        # Date Range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                datetime.now() - timedelta(days=90)
            )
        with col2:
            end_date = st.date_input(
                "End",
                datetime.now()
            )

        # Fetch catalyst tickers if selected
        if CATALYST_TYPES[catalyst] is not None:
            with st.spinner(f"Finding {catalyst} stocks..."):
                catalyst_tickers = fetch_catalyst_tickers(
                    CATALYST_TYPES[catalyst],
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                if catalyst_tickers:
                    tickers = list(set(tickers + catalyst_tickers))
                    st.success(f"Found {len(catalyst_tickers)} stocks with {catalyst}")
                    with st.expander("View catalyst tickers"):
                        st.write(", ".join(catalyst_tickers))

        # Show final ticker count
        if tickers:
            st.caption(f"Total tickers: {len(tickers)}")

        # Setup Selection
        st.subheader("Setups")
        all_setups = ['ORB 3m', 'ORB 5m', 'ORB 15m', 'ORB 60m',
                      'Episodic Pivots', 'Delayed Reactions', 'Earnings Plays']
        selected_setups = st.multiselect(
            "Select setups to test",
            all_setups,
            default=['ORB 15m', 'Episodic Pivots']
        )

        # Capital
        st.subheader("Capital")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        # Run Button
        st.markdown("---")
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Main Content
    if run_button:
        if not tickers:
            st.error("Please select tickers or choose a catalyst type to auto-populate tickers.")
        else:
            with st.spinner("Running backtest..."):
                result = run_backtest_with_progress(
                    tickers, start_date, end_date, selected_setups
                )

                if result and result[0] is not None:
                    trades_df, equity_df = result
                    st.session_state['trades_df'] = trades_df
                    st.session_state['equity_df'] = equity_df

                    # Generate pooled results if multiple tickers
                    if len(tickers) > 1:
                        st.session_state['pooled_results'] = generate_pooled_results(trades_df)

    # Display Results
    if 'trades_df' in st.session_state and not st.session_state['trades_df'].empty:
        trades_df = st.session_state['trades_df']

        # Key Metrics Row
        st.markdown("### 📊 Performance Summary")

        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()

        wins_sum = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losses_sum = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = wins_sum / losses_sum if losses_sum > 0 else float('inf')

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1%}")
        with col3:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        with col4:
            delta_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total PnL", f"${total_pnl:,.2f}", delta_color=delta_color)
        with col5:
            returns = (total_pnl / config.INITIAL_CAPITAL) * 100
            st.metric("Return", f"{returns:.1f}%")

        # Pooled Results (for multi-ticker analysis)
        if 'pooled_results' in st.session_state and st.session_state['pooled_results']:
            pooled = st.session_state['pooled_results']
            st.markdown("---")
            st.markdown("### 🔬 Pooled Analysis")
            st.markdown(f"*Aggregated results across {pooled.get('total_tickers', 0)} stocks*")

            # Recommendations
            for rec in pooled.get('recommendations', []):
                st.info(rec)

            # Best setups table
            if pooled.get('best_setups'):
                st.markdown("**Top Performing Setups:**")
                best_df = pd.DataFrame(pooled['best_setups'])
                st.dataframe(
                    best_df.style.format({
                        'win_rate': '{:.0%}',
                        'profit_factor': '{:.2f}',
                        'total_pnl': '${:,.0f}',
                        'avg_r': '{:.2f}R'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        # Charts Row 1
        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(create_equity_chart(trades_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_pnl_distribution(trades_df), use_container_width=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_setup_comparison(trades_df), use_container_width=True)

        with col2:
            st.plotly_chart(create_context_heatmap(trades_df), use_container_width=True)

        # Golden Setups
        st.markdown("### 🏆 Golden Setups")
        st.markdown("*Best performing condition combinations*")

        try:
            analyzer = TradeAnalyzer(trades_df)
            golden = analyzer.find_golden_setups(min_trades=3)

            if golden:
                for i, gs in enumerate(golden[:5], 1):
                    conditions = ', '.join([f"**{k}**={v}" for k, v in gs['conditions'].items()])
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.markdown(f"{i}. {conditions}")
                    with col2:
                        st.write(f"Trades: {gs['trades']}")
                    with col3:
                        st.write(f"WR: {gs['win_rate']:.0%}")
                    with col4:
                        st.write(f"PF: {gs['profit_factor']:.2f}")
            else:
                st.info("Not enough data to identify golden setups")
        except Exception as e:
            st.warning(f"Could not analyze golden setups: {e}")

        # Trade Log
        st.markdown("---")
        st.markdown("### 📋 Trade Log")

        display_cols = ['ticker', 'setup_type', 'entry_time', 'entry_price',
                        'exit_price', 'pnl', 'pnl_pct', 'r_multiple', 'exit_reason']
        display_cols = [c for c in display_cols if c in trades_df.columns]

        st.dataframe(
            trades_df[display_cols].style.format({
                'entry_price': '${:.2f}',
                'exit_price': '${:.2f}',
                'pnl': '${:.2f}',
                'pnl_pct': '{:.1f}%',
                'r_multiple': '{:.2f}R'
            }).applymap(
                lambda x: 'color: #22c55e' if isinstance(x, (int, float)) and x > 0 else 'color: #ef4444' if isinstance(x, (int, float)) and x < 0 else '',
                subset=['pnl', 'pnl_pct', 'r_multiple']
            ),
            use_container_width=True,
            height=400
        )

        # Download Button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log (CSV)",
            data=csv,
            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        # Welcome screen
        st.markdown("---")
        st.markdown("""
        ### Welcome! 👋

        This tool backtests **momentum trading setups** using historical market data from Polygon.io.

        **Available Setups:**
        - **ORB (Opening Range Breakout)** - Entry on breakout above opening range high
        - **Episodic Pivots** - Gap >4%, new highs, volume surge
        - **Delayed Reactions** - Post-EP consolidation breakouts
        - **Earnings Plays** - Post-earnings momentum trades

        **How to use:**
        1. Select your ticker watchlist in the sidebar
        2. Choose date range and setups to test
        3. Click **Run Backtest**
        4. Analyze results and find your golden setups!

        ---
        *Configure your backtest in the sidebar and click Run to get started.*
        """)


if __name__ == "__main__":
    main()
