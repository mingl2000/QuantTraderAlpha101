
import pandas as pd
import numpy as np
import backtrader as bt
import os
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from worldquant_101 import WorldQuant_101_Alphas
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from worldquant_101 import WorldQuant_101_Alphas

# Global defaults
DEFAULT_DATA_DIR_BASE = '/Volumes/IUSB/vipdoc/TDX'

def get_tickers(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('._')]
    # Ticker is filename without extension
    tickers = [f.replace('.csv', '') for f in files]
    tickers.sort()
    return tickers

def prepare_data_for_alphas(tickers, data_dir, start_date='2020-01-01'):
    # Dictionary to hold dataframes for each ticker temporarily
    ticker_dfs = {}
    
    print(f"Loading data from {data_dir} for {len(tickers)} tickers starting from {start_date}...")
    
    for ticker in tickers:
        filepath = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # Filter by date
            df = df[df.index >= pd.Timestamp(start_date)]
            
            if df.empty: continue
            
            # Standardize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            # Fix zero volume
            if 'volume' in df.columns:
                df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
                
            # Calculate Amount if missing
            if 'amount' not in df.columns:
                 df['amount'] = df['close'] * df['volume']
            
            ticker_dfs[ticker] = df
            
        except Exception as e:
            print(f"Error reading {ticker}: {e}")

    print(f"Data loaded. Constructing panels...")
    
    panel = {}
    
    if not ticker_dfs:
        print("No valid data found.")
        return panel

    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'returns', 'vwap']:
        field_data = {}
        for ticker, df in ticker_dfs.items():
            if field == 'returns':
                 if 'close' in df.columns:
                      field_data[ticker] = df['close'].pct_change()
            elif field in df.columns:
                field_data[ticker] = df[field]
        
        if field_data:
            panel[field] = pd.DataFrame(field_data)
        else:
             panel[field] = pd.DataFrame()
             
    return panel

def calculate_all_alphas(panel_data, DATASET_NAME):
    wq = WorldQuant_101_Alphas(panel_data)
    alphas = {}
    
    print("Calculating Alphas...")
    # Calculate all 101 alphas
    # for A500
    if DATASET_NAME =="A500":
        #alpha_102 (2/12) 172.83% # 4 days max top 10
        target_alphas = ['alpha_102','alpha_103','alpha_104','alpha_105','alpha_060','alpha_008', 'alpha_057', 'alpha_039', 'alpha_019', 'alpha_095', 'alpha_083', 'alpha_042']
        #target_alphas =['alpha_060']
        #target_alphas =['alpha_102']
    #for A1000
    elif DATASET_NAME =="A1000":
        target_alphas = ['alpha_031','alpha_009','alpha_011','alpha_037','alpha_060','alpha_083','alpha_017','alpha_053','alpha_081','alpha_052']
        #target_alphas =['alpha_102']
    #for HS300
    elif DATASET_NAME =="HS300":
        target_alphas = ['alpha_083','alpha_060','alpha_008','alpha_033','alpha_052','alpha_028','alpha_025','alpha_019','alpha_005','alpha_095']
    elif DATASET_NAME =="AAll":
        target_alphas = ['alpha_083','alpha_060','alpha_008','alpha_033','alpha_052','alpha_028','alpha_025','alpha_019','alpha_005','alpha_095','alpha_060','alpha_008', 'alpha_057', 'alpha_039', 'alpha_019', 'alpha_095', 'alpha_083', 'alpha_042','alpha_031','alpha_009','alpha_011','alpha_037','alpha_060','alpha_083','alpha_017','alpha_053','alpha_081','alpha_052']
        target_alphas =list(set(target_alphas))
        target_alphas=['alpha_052','alpha_095','alpha_011','alpha_081','alpha_009','alpha_028','alpha_031','alpha_042','alpha_060','alpha_025']
        #target_alphas =['alpha_102']
    elif DATASET_NAME =="KCCY50":
        target_alphas = ['alpha_026','alpha_024','alpha_040','alpha_083','alpha_099','alpha_005','alpha_075','alpha_032','alpha_004','alpha_077']
    elif DATASET_NAME =="KC50":
        target_alphas = ['alpha_083','alpha_060','alpha_008','alpha_033','alpha_052','alpha_028','alpha_025','alpha_019','alpha_005','alpha_095']
    else:target_alphas = [f"alpha_{i:03d}" for i in range(1, 102)]

    for alpha_name in target_alphas:
        if hasattr(wq, alpha_name):
            try:
                alpha_func = getattr(wq, alpha_name)
                signal = alpha_func()
                
                if signal is not None and not signal.empty:
                    if not (signal == 0).all().all():
                        alphas[alpha_name] = signal
                        print(f"Calculated {alpha_name}   ", end='\r')
            except Exception as e:
                pass
                
    return alphas

class WQAlphaStrategy(bt.Strategy):
    params = (('signal', None), ('top_n', 10), ('save_picks', False))
    
    def __init__(self):
        self.signal = self.params.signal
        self.daily_picks = [] 
        self.equity_curve = []
        
    def next(self):
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        
        self.equity_curve.append({'Date': dt, 'Value': self.broker.getvalue()})
        
        if dt_ts not in self.signal.index:
            return
            
        daily_signal = self.signal.loc[dt_ts]
        present_tickers = [d._name for d in self.datas if len(d) > 0]
        if not present_tickers: return
        
        valid_tickers = [t for t in present_tickers if t in daily_signal.index]
        valid_signals = daily_signal[valid_tickers].dropna()
        if valid_signals.empty: return
        
        if (valid_signals == 0).all():
            longs = []
        else:
            sorted_signals = valid_signals.sort_values(ascending=False)
            target_n = self.params.top_n
            longs = sorted_signals.head(target_n).index.tolist()
        
        if self.params.save_picks:
            try:
                self.daily_picks.append({'Date': dt, 'Picks': ", ".join(longs)})
            except: pass
        
        weight = 0.95 / len(longs) if longs else 0
        current_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        dropouts = [t for t in current_positions if t not in longs]
        
        for d in self.datas:
            if d._name in longs:
                self.order_target_percent(d, target=weight)
            elif d._name in dropouts:
                self.order_target_percent(d, target=0.0)
            else:
                if self.getposition(d).size != 0:
                    self.order_target_percent(d, target=0.0)

    def notify_order(self, order):
        pass

detailed_results = {}

def run_single_backtest(saved_files, alpha_signal, alpha_name, detailed=False):
    if isinstance(alpha_signal.index, pd.DatetimeIndex) and alpha_signal.index.tz is not None:
        alpha_signal.index = alpha_signal.index.tz_localize(None)
    
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(WQAlphaStrategy, signal=alpha_signal, top_n=10, save_picks=detailed)
    
    data_added = False
    valid_tickers = [t for t in alpha_signal.columns]
    
    for filepath in saved_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace('.csv', '')
        
        if ticker not in valid_tickers: continue
        if alpha_signal[ticker].dropna().empty: continue
            
        data = bt.feeds.GenericCSVData(
            dataname=filepath, fromdate=pd.Timestamp('2020-01-01'), todate=None, dtformat='%Y-%m-%d',
            datetime=0, open=1, high=2, low=3, close=4, volume=6, openinterest=-1,
            plot=False
        )
        cerebro.adddata(data, name=ticker)
        data_added = True
        
    if not data_added:
        return 0.0
        
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    initial_value = cerebro.broker.getvalue()
    strategies = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(type(strategies))
    print(type(detailed))
    if detailed and strategies:
        strat = strategies[0]
        detailed_results['daily_picks'] = strat.daily_picks
        detailed_results['equity_curve'] = strat.equity_curve
    
    
    
        # 获取分析器结果
        try:
            sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            if isinstance(sharpe_ratio, dict):
                sharpe_ratio = sharpe_ratio.get('sharperatio', 0)
            max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            
            # 打印结果
            print(f"初始资金: {initial_value:.2f}")
            print(f"最终资金: {final_value:.2f}")
            
            print(f"夏普比率: {sharpe_ratio:.4f}")
            print(f"最大回撤: {max_drawdown:.2%}")
            
            # 获取交ç易分析
            trade_analysis = strat.analyzers.trades.get_analysis()
            print("111 trade_analysis:", type(trade_analysis))
            # 获取胜率
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            print("222")
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            print("333")
            win_rate = won_trades / total_trades if total_trades > 0 else 0
            print("444")
            if win_rate is not null:
                print(f"胜率: {win_rate:.2%}")
            # 获取平均收益
            won_pnl = trade_analysis.get('won', {}).get('pnl', 0)   
            print("555")
            lost_pnl = trade_analysis.get('lost', {}).get('pnl', 0)

            print("666")
            '''
            if won_pnl is not None and won_trades is not None:
                avg_won = won_pnl / won_trades if won_trades > 0 else 0
                print(f"平均盈利: {avg_won:.2f}")
            print("777")
            if  lost_pnl is not null and total_trades is not null and won_trades is not null:
                avg_lost = lost_pnl / (total_trades - won_trades) if (total_trades - won_trades) > 0 else 0
                print(f"平均亏损: {avg_lost:.2f}")
                print(f"交易次数: {total_trades}")
            print("888")
            
            
            
            '''
        
            
            # 绘制结果
            print("999")
            cerebro.plot(style='candle', figsize=figsize)
            plt.show()
            print("aaa")
        except Exception as e:
            print(f"Error in backtesting: {alpha_name}") 
        
    ret = (final_value - initial_value) / initial_value
    print(f"总收益率: {ret:.2%}")
    return ret

def main():
    parser = argparse.ArgumentParser(description='Backtest WQ101 Alphas on Generic Dataset')
    parser.add_argument('--dataset', type=str, default='A500', help='Dataset folder name in TDX/ (e.g. A500, A1000)')
    parser.add_argument('--index_id', type=str, default=None, help='Index file ID (e.g. 000510, 000852). Optional.')
    args = parser.parse_args()
    
    DATASET_NAME = args.dataset
    DATA_DIR = os.path.join(DEFAULT_DATA_DIR_BASE, DATASET_NAME)
    
    # Auto-detect index ID if not provided
    INDEX_ID = args.index_id
    if INDEX_ID is None:
        if DATASET_NAME == 'A500':
            INDEX_ID = '000510'
        elif DATASET_NAME == 'A1000':
            INDEX_ID = '000852'
        else:
            # Default fallback? Or None
            INDEX_ID = None
            
    print(f"--- Running Backtest for {DATASET_NAME} ---")
    
    # Get Tickers
    tickers = get_tickers(DATA_DIR)
    if not tickers:
        print(f"No tickers found in {DATA_DIR}. Exiting.")
        sys.exit(1)
        
    print(f"Found {len(tickers)} tickers in {DATA_DIR}.")
    
    # Prepare Data
    panel_data = prepare_data_for_alphas(tickers, DATA_DIR)
    
    # Calculate Alphas
    all_alphas = calculate_all_alphas(panel_data, DATASET_NAME)
    print(f"Computed {len(all_alphas)} valid alpha signals.")
    
    # --- Index Regime Filter ---
    INDEX_FILE = None
    if INDEX_ID:
        INDEX_FILE = os.path.join(DEFAULT_DATA_DIR_BASE, f"{INDEX_ID}.csv")
    
    if INDEX_FILE and os.path.exists(INDEX_FILE):
        print(f"Loading Index Data from {INDEX_FILE}...")
        try:
            df_index = pd.read_csv(INDEX_FILE, index_col=0, parse_dates=True)
            print(df_index.head())
            index_amount = df_index['Amount'] #bese for A500 1.31%
            index_close = df_index['Close']
            index_volume = df_index['Volume']
            #index_close = df_index['Vwap']
            
            # Calculate OBV
            obv = (np.sign(index_close.diff()) * index_volume).fillna(0).cumsum()
            
            #index_ma60 = index_close.ewm(span=60, adjust=False).mean() #113.9%
            #index_ma20 = index_close.ewm(span=20, adjust=False).mean()
            index_ma60 = index_amount.rolling(60).mean()#131.62%
            index_ma10 = index_amount.rolling(10).mean()
            index_ma5 = index_amount.rolling(5).mean() #alpha_060 141.1%
            index_ma2 = index_amount.rolling(2).mean() #alpha_060 141.1%


            #regime = ((index_close > index_ma60) & (index_ma60>index_ma60.shift(1))& (index_ma60>index_ma20)).astype(int)
            #regime = ((index_close > index_ma60) & (index_ma20>index_ma60)).astype(int)
            #regime = ((index_ma10 > index_ma60)).astype(int) #alpha_083 1.371106
            regime = ((index_ma2> index_ma60)).astype(int) #alpha_060 141.1%
            
            #regime = True
            print("Applying Regime Filter (Close > MA60)...")
            filtered_alphas = {}
            for name, signal in all_alphas.items():
                aligned_regime = regime.reindex(signal.index).fillna(method='ffill')
                filtered_signal = signal.mul(aligned_regime, axis=0)
                filtered_alphas[name] = filtered_signal
            
            all_alphas = filtered_alphas
            print("Regime Filter Applied.")
        except Exception as e:
            print(f"Error applying regime filter: {e}")
    else:
        if INDEX_ID:
            print(f"Warning: Index file {INDEX_FILE} not found. Running WITHOUT regime filter.")
        else:
             print("No index ID provided/detected. Running WITHOUT regime filter.")
    # ---------------------------
    
    saved_files = [os.path.join(DATA_DIR, f"{t}.csv") for t in tickers]
    results = []
    
    print("Running Backtests (Top 10 Daily Rebalance)...")
    for idx, (alpha_name, signal) in enumerate(all_alphas.items()):
        print(f"Running {alpha_name} ({idx+1}/{len(all_alphas)})...", end='\r')
        try:
            ret = run_single_backtest(saved_files, signal, alpha_name)
            results.append({'Alpha': alpha_name, 'Return': ret})
        except Exception as e:
            print(f"Backtest failed for {alpha_name}: {e}")
            
    print("\nDone.")
    
    df_results = pd.DataFrame(results)
    print(df_results.columns)
    df_results = df_results.sort_values(by='Return', ascending=False)
    
    output_csv = f'WQ101_Backtest_Results_{DATASET_NAME}.csv'
    df_results.to_csv(output_csv, index=False)
    
    print("-" * 40)
    print(f"Top 10 Alphas ({DATASET_NAME}):")
    print(df_results.head(10).to_string(index=False))
    print("-" * 40)
    print(f"Results saved to {output_csv}")
    
    # Detailed Analysis for Top 10 Alphas
    if not df_results.empty:
        top_n_alphas = df_results.head(10)['Alpha'].tolist()
        print(f"\nGenerating detailed analysis for Top {len(top_n_alphas)} Alphas...")
        
        for alpha_name in top_n_alphas:
            print(f"Processing {alpha_name}...", end='\r')
            detailed_results.clear()
            
            alpha_signal = all_alphas[alpha_name]
            run_single_backtest(saved_files, alpha_signal, alpha_name, detailed=True)
            
            if 'daily_picks' in detailed_results:
                picks_df = pd.DataFrame(detailed_results['daily_picks'])
                if not picks_df.empty:
                    picks_file = f"{alpha_name}_daily_picks_{DATASET_NAME}.csv"
                    picks_df.to_csv(picks_file, index=False)
            
            if 'equity_curve' in detailed_results:
                try:
                    equity_df = pd.DataFrame(detailed_results['equity_curve'])
                    if not equity_df.empty:
                        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
                        equity_df = equity_df.set_index('Date')
                        equity_df['Growth'] = equity_df['Value'] / equity_df['Value'].iloc[0]
                        
                        plt.figure(figsize=(12, 6))
                        final_return = df_results[df_results['Alpha'] == alpha_name]['Return'].values[0]
                        plt.axes().set_title(f"Cumulative Return ({DATASET_NAME}) - {alpha_name}: {final_return:.2%}")
                        
                        equity_df['Growth'].plot(label='Strategy')
                        plt.xlabel("Date")
                        plt.ylabel("Growth of $1")
                        plt.grid(True)
                        plt.legend()
                        
                        plot_file = f"{alpha_name}_equity_curve_{DATASET_NAME}.png"
                        plt.savefig(plot_file)
                        plt.close()
                except Exception as e:
                    print(f"Plotting failed for {alpha_name}: {e}")
                    
        print(f"\nAnalysis Complete. Plots and picks saved for top {len(top_n_alphas)} alphas.")

if __name__ == '__main__':
    main()
