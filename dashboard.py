"""
Plotly Dashboard for Search Arbitrage Backtest Results.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any
import os
import config


class Dashboard:
    """
    Generates interactive Plotly dashboards for backtest visualization.
    """

    def __init__(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame = None):
        self.trades_df = trades_df
        self.equity_df = equity_df
        self.colors = {
            'green': '#22c55e',
            'red': '#ef4444',
            'blue': '#3b82f6',
            'purple': '#8b5cf6',
            'orange': '#f97316',
            'cyan': '#06b6d4',
            'gray': '#6b7280',
            'bg': '#1a1a2e',
            'card': '#16213e',
            'text': '#e5e7eb'
        }

    def create_full_dashboard(self, output_path: str = None) -> go.Figure:
        """Create comprehensive dashboard with all visualizations."""

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Equity Curve', 'PnL Distribution',
                'Setup Performance', 'Win Rate by Market Regime',
                'Best Contexts Heatmap', 'R-Multiple Distribution',
                'Monthly Returns', 'Cumulative Trades by Setup'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        # 1. Equity Curve
        self._add_equity_curve(fig, row=1, col=1)

        # 2. PnL Distribution
        self._add_pnl_distribution(fig, row=1, col=2)

        # 3. Setup Performance
        self._add_setup_performance(fig, row=2, col=1)

        # 4. Win Rate by Market Regime
        self._add_market_regime_chart(fig, row=2, col=2)

        # 5. Best Contexts Heatmap
        self._add_context_heatmap(fig, row=3, col=1)

        # 6. R-Multiple Distribution
        self._add_r_multiple_dist(fig, row=3, col=2)

        # 7. Monthly Returns
        self._add_monthly_returns(fig, row=4, col=1)

        # 8. Cumulative Trades
        self._add_cumulative_trades(fig, row=4, col=2)

        # Update layout
        fig.update_layout(
            height=1600,
            width=1400,
            title_text="Search Arbitrage Backtest Dashboard",
            title_font_size=24,
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor=self.colors['bg'],
            plot_bgcolor=self.colors['card'],
            font=dict(color=self.colors['text'])
        )

        if output_path:
            fig.write_html(output_path)
            print(f"Dashboard saved to {output_path}")

        return fig

    def _add_equity_curve(self, fig: go.Figure, row: int, col: int):
        """Add equity curve plot."""
        if self.equity_df is None or self.equity_df.empty:
            # Create from trades if no equity df
            if not self.trades_df.empty and 'entry_time' in self.trades_df.columns:
                cumulative = self.trades_df.sort_values('entry_time')['pnl'].cumsum()
                cumulative = cumulative + config.INITIAL_CAPITAL

                fig.add_trace(
                    go.Scatter(
                        x=self.trades_df.sort_values('entry_time')['entry_time'],
                        y=cumulative,
                        mode='lines',
                        name='Equity',
                        line=dict(color=self.colors['green'], width=2)
                    ),
                    row=row, col=col
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=self.equity_df['date'],
                    y=self.equity_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color=self.colors['green'], width=2)
                ),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Equity ($)", row=row, col=col)

    def _add_pnl_distribution(self, fig: go.Figure, row: int, col: int):
        """Add PnL distribution histogram."""
        if self.trades_df.empty:
            return

        winners = self.trades_df[self.trades_df['pnl'] > 0]['pnl']
        losers = self.trades_df[self.trades_df['pnl'] <= 0]['pnl']

        fig.add_trace(
            go.Histogram(
                x=winners,
                name='Winners',
                marker_color=self.colors['green'],
                opacity=0.7,
                nbinsx=30
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Histogram(
                x=losers,
                name='Losers',
                marker_color=self.colors['red'],
                opacity=0.7,
                nbinsx=30
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="PnL ($)", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_setup_performance(self, fig: go.Figure, row: int, col: int):
        """Add setup performance comparison."""
        if self.trades_df.empty or 'setup_type' not in self.trades_df.columns:
            return

        setup_stats = []
        for setup in self.trades_df['setup_type'].unique():
            subset = self.trades_df[self.trades_df['setup_type'] == setup]
            wins = subset[subset['pnl'] > 0]['pnl'].sum()
            losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
            pf = wins / losses if losses > 0 else (5 if wins > 0 else 0)

            setup_stats.append({
                'setup': setup,
                'profit_factor': min(pf, 5),  # Cap at 5 for visualization
                'win_rate': (subset['pnl'] > 0).mean() * 100,
                'trades': len(subset)
            })

        df = pd.DataFrame(setup_stats)

        fig.add_trace(
            go.Bar(
                x=df['setup'],
                y=df['profit_factor'],
                name='Profit Factor',
                marker_color=self.colors['blue'],
                text=df['profit_factor'].round(2),
                textposition='auto'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Setup", tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text="Profit Factor", row=row, col=col)

    def _add_market_regime_chart(self, fig: go.Figure, row: int, col: int):
        """Add win rate by market regime."""
        regime_col = 'ctx_market_regime'
        if self.trades_df.empty or regime_col not in self.trades_df.columns:
            return

        regime_stats = []
        for regime in self.trades_df[regime_col].dropna().unique():
            subset = self.trades_df[self.trades_df[regime_col] == regime]
            regime_stats.append({
                'regime': regime,
                'win_rate': (subset['pnl'] > 0).mean() * 100,
                'trades': len(subset)
            })

        df = pd.DataFrame(regime_stats)

        colors = [self.colors['green'] if r == 'Risk_On' else
                  self.colors['red'] if r == 'Risk_Off' else
                  self.colors['orange'] for r in df['regime']]

        fig.add_trace(
            go.Bar(
                x=df['regime'],
                y=df['win_rate'],
                name='Win Rate',
                marker_color=colors,
                text=[f"{w:.1f}%" for w in df['win_rate']],
                textposition='auto'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Market Regime", row=row, col=col)
        fig.update_yaxes(title_text="Win Rate (%)", row=row, col=col)

    def _add_context_heatmap(self, fig: go.Figure, row: int, col: int):
        """Add heatmap of context variables vs profit factor."""
        if self.trades_df.empty:
            return

        # Find categorical context columns
        ctx_cols = [col for col in self.trades_df.columns
                    if col.startswith('ctx_') and self.trades_df[col].dtype == 'object']

        if not ctx_cols:
            return

        # Build heatmap data
        heatmap_data = []
        x_labels = []
        y_labels = []

        for col in ctx_cols[:5]:  # Limit to 5 variables
            var_name = col.replace('ctx_', '')
            y_labels.append(var_name)

            row_data = []
            values = self.trades_df[col].dropna().unique()[:5]  # Limit values

            for val in values:
                if var_name == y_labels[0]:  # Only add x labels once
                    x_labels.append(str(val)[:15])  # Truncate long labels

                subset = self.trades_df[self.trades_df[col] == val]
                if len(subset) >= 3:
                    wins = subset[subset['pnl'] > 0]['pnl'].sum()
                    losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
                    pf = wins / losses if losses > 0 else 3
                    row_data.append(min(pf, 3))
                else:
                    row_data.append(np.nan)

            # Pad if needed
            while len(row_data) < len(x_labels):
                row_data.append(np.nan)
            heatmap_data.append(row_data[:len(x_labels)])

        if heatmap_data:
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=x_labels,
                    y=y_labels,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='PF')
                ),
                row=row, col=col
            )

    def _add_r_multiple_dist(self, fig: go.Figure, row: int, col: int):
        """Add R-multiple distribution."""
        if self.trades_df.empty or 'r_multiple' not in self.trades_df.columns:
            return

        r_values = self.trades_df['r_multiple'].dropna()

        fig.add_trace(
            go.Histogram(
                x=r_values,
                name='R-Multiple',
                marker_color=self.colors['purple'],
                nbinsx=40
            ),
            row=row, col=col
        )

        # Add vertical lines at key levels
        for r_level, color in [(0, 'white'), (1, self.colors['green']), (-1, self.colors['red'])]:
            fig.add_vline(
                x=r_level, line_dash="dash", line_color=color,
                row=row, col=col
            )

        fig.update_xaxes(title_text="R-Multiple", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_monthly_returns(self, fig: go.Figure, row: int, col: int):
        """Add monthly returns bar chart."""
        if self.trades_df.empty or 'entry_time' not in self.trades_df.columns:
            return

        df = self.trades_df.copy()
        df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')

        monthly = df.groupby('month')['pnl'].sum()

        colors = [self.colors['green'] if v >= 0 else self.colors['red']
                  for v in monthly.values]

        fig.add_trace(
            go.Bar(
                x=[str(m) for m in monthly.index],
                y=monthly.values,
                name='Monthly PnL',
                marker_color=colors
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Month", tickangle=45, row=row, col=col)
        fig.update_yaxes(title_text="PnL ($)", row=row, col=col)

    def _add_cumulative_trades(self, fig: go.Figure, row: int, col: int):
        """Add cumulative trade count by setup."""
        if self.trades_df.empty or 'entry_time' not in self.trades_df.columns:
            return

        df = self.trades_df.sort_values('entry_time')

        for setup in df['setup_type'].unique():
            setup_df = df[df['setup_type'] == setup]
            cumulative = range(1, len(setup_df) + 1)

            fig.add_trace(
                go.Scatter(
                    x=setup_df['entry_time'],
                    y=cumulative,
                    mode='lines',
                    name=setup
                ),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Cumulative Trades", row=row, col=col)

    def create_golden_setup_chart(self, golden_setups: List[Dict]) -> go.Figure:
        """Create visualization of golden setups."""
        if not golden_setups:
            return go.Figure()

        # Prepare data
        labels = []
        pfs = []
        win_rates = []
        trade_counts = []

        for gs in golden_setups[:10]:
            label = '\n'.join([f"{k}={v}" for k, v in gs['conditions'].items()])
            labels.append(label[:40])  # Truncate
            pfs.append(gs['profit_factor'])
            win_rates.append(gs['win_rate'] * 100)
            trade_counts.append(gs['trades'])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Profit Factor', 'Win Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(
                y=labels,
                x=pfs,
                orientation='h',
                marker_color=self.colors['green'],
                text=[f"{pf:.2f}" for pf in pfs],
                textposition='outside'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                y=labels,
                x=win_rates,
                orientation='h',
                marker_color=self.colors['blue'],
                text=[f"{wr:.1f}%" for wr in win_rates],
                textposition='outside'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            title_text="Golden Setups - Best Performing Combinations",
            template="plotly_dark",
            showlegend=False
        )

        return fig


def generate_dashboard(
    trades_csv: str,
    equity_csv: str = None,
    output_dir: str = config.OUTPUT_DIR
) -> str:
    """
    Generate dashboard from saved CSV files.

    Args:
        trades_csv: Path to master trade log CSV
        equity_csv: Path to equity curve CSV (optional)
        output_dir: Output directory for dashboard

    Returns:
        Path to generated dashboard HTML
    """
    trades_df = pd.read_csv(trades_csv)

    equity_df = None
    if equity_csv and os.path.exists(equity_csv):
        equity_df = pd.read_csv(equity_csv)

    dashboard = Dashboard(trades_df, equity_df)

    output_path = os.path.join(output_dir, config.DASHBOARD_FILE)
    dashboard.create_full_dashboard(output_path)

    return output_path
