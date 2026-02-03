"""
Analyzer Module.
Performs correlation analysis and generates insights from backtest results.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
import config


class TradeAnalyzer:
    """
    Analyzes completed trades to find optimal conditions.
    """

    def __init__(self, trades_df: pd.DataFrame):
        self.trades_df = trades_df
        self.context_cols = [col for col in trades_df.columns if col.startswith("ctx_")]
        self.meta_cols = [col for col in trades_df.columns if col.startswith("meta_")]

    def analyze_all(self) -> Dict[str, Any]:
        """Run all analyses and return comprehensive report."""
        return {
            "best_contexts": self.find_best_contexts(),
            "worst_contexts": self.find_worst_contexts(),
            "context_correlations": self.get_pnl_correlations(),
            "setup_comparison": self.compare_setups(),
            "timeframe_analysis": self.analyze_orb_timeframes(),
            "market_regime_impact": self.analyze_market_regime(),
            "golden_setups": self.find_golden_setups()
        }

    def find_best_contexts(self, min_trades: int = 5, min_pf: float = 2.0) -> pd.DataFrame:
        """
        Find context conditions that correlate with high profit factors.
        """
        results = []

        for col in self.context_cols + self.meta_cols:
            col_data = self.trades_df[col]

            if col_data.isna().all():
                continue

            if col_data.dtype == 'object' or col_data.nunique() < 10:
                # Categorical analysis
                for value in col_data.dropna().unique():
                    subset = self.trades_df[col_data == value]
                    if len(subset) >= min_trades:
                        stats = self._calc_subset_stats(subset)
                        if stats['profit_factor'] >= min_pf:
                            results.append({
                                'variable': col.replace('ctx_', '').replace('meta_', ''),
                                'condition': f"= {value}",
                                **stats
                            })
            else:
                # Numeric - analyze quantiles
                try:
                    quartiles = pd.qcut(col_data.dropna(), q=4, duplicates='drop')
                    for q in quartiles.unique():
                        subset = self.trades_df[quartiles == q]
                        if len(subset) >= min_trades:
                            stats = self._calc_subset_stats(subset)
                            if stats['profit_factor'] >= min_pf:
                                results.append({
                                    'variable': col.replace('ctx_', '').replace('meta_', ''),
                                    'condition': str(q),
                                    **stats
                                })
                except (ValueError, TypeError):
                    continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('profit_factor', ascending=False)
        return df

    def find_worst_contexts(self, min_trades: int = 5, max_pf: float = 0.5) -> pd.DataFrame:
        """
        Find context conditions that correlate with poor performance.
        Useful for filtering out bad setups.
        """
        results = []

        for col in self.context_cols + self.meta_cols:
            col_data = self.trades_df[col]

            if col_data.isna().all():
                continue

            if col_data.dtype == 'object' or col_data.nunique() < 10:
                for value in col_data.dropna().unique():
                    subset = self.trades_df[col_data == value]
                    if len(subset) >= min_trades:
                        stats = self._calc_subset_stats(subset)
                        if stats['profit_factor'] <= max_pf:
                            results.append({
                                'variable': col.replace('ctx_', '').replace('meta_', ''),
                                'condition': f"= {value}",
                                **stats
                            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('profit_factor', ascending=True)
        return df

    def get_pnl_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between numeric context variables and PnL.
        """
        numeric_cols = []
        for col in self.context_cols + self.meta_cols:
            if self.trades_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                numeric_cols.append(col)

        if not numeric_cols:
            return pd.DataFrame()

        correlations = []
        for col in numeric_cols:
            valid_data = self.trades_df[[col, 'pnl']].dropna()
            if len(valid_data) > 5:
                corr, pvalue = stats.pearsonr(valid_data[col], valid_data['pnl'])
                correlations.append({
                    'variable': col.replace('ctx_', '').replace('meta_', ''),
                    'correlation': corr,
                    'p_value': pvalue,
                    'significant': pvalue < 0.05
                })

        df = pd.DataFrame(correlations)
        if not df.empty:
            df = df.sort_values('correlation', ascending=False, key=abs)
        return df

    def compare_setups(self) -> pd.DataFrame:
        """Compare performance across all setup types."""
        results = []

        for setup_type in self.trades_df['setup_type'].unique():
            subset = self.trades_df[self.trades_df['setup_type'] == setup_type]
            stats = self._calc_subset_stats(subset)
            stats['setup_type'] = setup_type
            results.append(stats)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('setup_type')
        return df

    def analyze_orb_timeframes(self) -> pd.DataFrame:
        """
        Analyze which ORB timeframe performs best based on ADR.
        """
        orb_trades = self.trades_df[
            self.trades_df['setup_type'].str.startswith('ORB_')
        ].copy()

        if orb_trades.empty:
            return pd.DataFrame()

        # Extract timeframe
        orb_trades['timeframe'] = orb_trades['setup_type'].str.extract(r'ORB_(\d+)m')[0].astype(int)

        # Analyze by timeframe and ADR bucket
        results = []

        adr_col = 'ctx_adr_pct' if 'ctx_adr_pct' in orb_trades.columns else None

        for tf in orb_trades['timeframe'].unique():
            tf_subset = orb_trades[orb_trades['timeframe'] == tf]

            base_stats = self._calc_subset_stats(tf_subset)
            base_stats['timeframe'] = f"{tf}m"
            base_stats['adr_bucket'] = 'All'
            results.append(base_stats)

            # If ADR data available, break down further
            if adr_col and not tf_subset[adr_col].isna().all():
                try:
                    tf_subset['adr_bucket'] = pd.qcut(
                        tf_subset[adr_col], q=3,
                        labels=['Low_ADR', 'Mid_ADR', 'High_ADR'],
                        duplicates='drop'
                    )
                    for bucket in tf_subset['adr_bucket'].unique():
                        bucket_subset = tf_subset[tf_subset['adr_bucket'] == bucket]
                        if len(bucket_subset) >= 3:
                            bucket_stats = self._calc_subset_stats(bucket_subset)
                            bucket_stats['timeframe'] = f"{tf}m"
                            bucket_stats['adr_bucket'] = bucket
                            results.append(bucket_stats)
                except (ValueError, TypeError):
                    pass

        return pd.DataFrame(results)

    def analyze_market_regime(self) -> pd.DataFrame:
        """Analyze performance by market regime."""
        regime_col = 'ctx_market_regime'
        if regime_col not in self.trades_df.columns:
            return pd.DataFrame()

        results = []
        for regime in self.trades_df[regime_col].dropna().unique():
            subset = self.trades_df[self.trades_df[regime_col] == regime]
            if len(subset) >= 3:
                stats = self._calc_subset_stats(subset)
                stats['market_regime'] = regime
                results.append(stats)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index('market_regime')
        return df

    def find_golden_setups(self, min_trades: int = 5) -> List[Dict]:
        """
        Find "Golden Setups" - combinations that yield exceptional results.

        A Golden Setup typically has:
        - Win rate > 60%
        - Profit Factor > 2.0
        - At least `min_trades` occurrences
        """
        golden_setups = []

        # Key context variables to combine
        key_vars = ['ctx_market_regime', 'ctx_sector', 'ctx_float_category',
                    'ctx_rvol_category', 'ctx_ma_alignment']

        # Filter to available columns
        available_vars = [v for v in key_vars if v in self.trades_df.columns]

        # Analyze single conditions first
        for var in available_vars:
            for value in self.trades_df[var].dropna().unique():
                subset = self.trades_df[self.trades_df[var] == value]
                if len(subset) >= min_trades:
                    stats = self._calc_subset_stats(subset)
                    if stats['win_rate'] > 0.6 and stats['profit_factor'] > 2.0:
                        golden_setups.append({
                            'conditions': {var.replace('ctx_', ''): value},
                            **stats
                        })

        # Two-variable combinations
        if len(available_vars) >= 2:
            for i, var1 in enumerate(available_vars):
                for var2 in available_vars[i + 1:]:
                    for val1 in self.trades_df[var1].dropna().unique():
                        for val2 in self.trades_df[var2].dropna().unique():
                            mask = (self.trades_df[var1] == val1) & (self.trades_df[var2] == val2)
                            subset = self.trades_df[mask]
                            if len(subset) >= min_trades:
                                stats = self._calc_subset_stats(subset)
                                if stats['win_rate'] > 0.65 and stats['profit_factor'] > 2.5:
                                    golden_setups.append({
                                        'conditions': {
                                            var1.replace('ctx_', ''): val1,
                                            var2.replace('ctx_', ''): val2
                                        },
                                        **stats
                                    })

        # Sort by profit factor
        golden_setups.sort(key=lambda x: x['profit_factor'], reverse=True)

        return golden_setups[:20]  # Top 20 golden setups

    def _calc_subset_stats(self, subset: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance statistics for a subset of trades."""
        if subset.empty:
            return {
                'trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_pnl': 0, 'total_pnl': 0, 'avg_r': 0
            }

        wins = subset[subset['pnl'] > 0]['pnl'].sum()
        losses = abs(subset[subset['pnl'] <= 0]['pnl'].sum())

        return {
            'trades': len(subset),
            'win_rate': (subset['pnl'] > 0).mean(),
            'profit_factor': wins / losses if losses > 0 else (float('inf') if wins > 0 else 0),
            'avg_pnl': subset['pnl'].mean(),
            'total_pnl': subset['pnl'].sum(),
            'avg_r': subset['r_multiple'].mean() if 'r_multiple' in subset.columns else 0,
            'max_win': subset['pnl'].max(),
            'max_loss': subset['pnl'].min()
        }

    def generate_report(self, output_path: str = None) -> str:
        """Generate a comprehensive text report."""
        analysis = self.analyze_all()

        report = []
        report.append("=" * 70)
        report.append("SEARCH ARBITRAGE BACKTEST ANALYSIS REPORT")
        report.append("=" * 70)

        # Setup Comparison
        report.append("\n\n### SETUP COMPARISON ###\n")
        if not analysis['setup_comparison'].empty:
            report.append(analysis['setup_comparison'].to_string())

        # Best Contexts
        report.append("\n\n### BEST PERFORMING CONTEXTS (PF > 2.0) ###\n")
        if not analysis['best_contexts'].empty:
            report.append(analysis['best_contexts'].head(15).to_string())

        # Worst Contexts
        report.append("\n\n### CONTEXTS TO AVOID (PF < 0.5) ###\n")
        if not analysis['worst_contexts'].empty:
            report.append(analysis['worst_contexts'].head(10).to_string())

        # ORB Timeframe Analysis
        report.append("\n\n### ORB TIMEFRAME ANALYSIS ###\n")
        if not analysis['timeframe_analysis'].empty:
            report.append(analysis['timeframe_analysis'].to_string())

        # Market Regime Impact
        report.append("\n\n### MARKET REGIME IMPACT ###\n")
        if not analysis['market_regime_impact'].empty:
            report.append(analysis['market_regime_impact'].to_string())

        # Golden Setups
        report.append("\n\n### GOLDEN SETUPS (Best Combinations) ###\n")
        for i, gs in enumerate(analysis['golden_setups'][:10], 1):
            conditions = ', '.join([f"{k}={v}" for k, v in gs['conditions'].items()])
            report.append(f"\n{i}. {conditions}")
            report.append(f"   Trades: {gs['trades']}, Win Rate: {gs['win_rate']:.1%}, "
                          f"PF: {gs['profit_factor']:.2f}, Avg R: {gs['avg_r']:.2f}")

        # Context Correlations
        report.append("\n\n### CONTEXT-PNL CORRELATIONS ###\n")
        if not analysis['context_correlations'].empty:
            report.append(analysis['context_correlations'].to_string())

        report_text = '\n'.join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")

        return report_text
