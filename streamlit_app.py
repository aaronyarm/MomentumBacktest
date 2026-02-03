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
    "Mega Tech": ["NVDA", "AMD", "TSLA", "META", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "MU", "MRVL", "QCOM", "INTC", "ASML"],
    "Software/SaaS": ["CRM", "NOW", "ADBE", "SNOW", "PLTR", "NET", "DDOG", "ZS"],
    "Fintech": ["SQ", "COIN", "PYPL", "HOOD", "SOFI", "AFRM"],
    "Custom": []
}


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
        progress_bar.progress(10, text="Loading market data...")

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
        st.error(f"Backtest error: {str(e)}")
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

        if preset == "Custom":
            ticker_input = st.text_area(
                "Enter tickers (comma separated)",
                "NVDA, AMD, TSLA, META"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",")]
        else:
            tickers = TICKER_PRESETS[preset]
            st.write(f"Selected: {', '.join(tickers)}")

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
        with st.spinner("Running backtest..."):
            result = run_backtest_with_progress(
                tickers, start_date, end_date, selected_setups
            )

            if result and result[0] is not None:
                trades_df, equity_df = result
                st.session_state['trades_df'] = trades_df
                st.session_state['equity_df'] = equity_df

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
