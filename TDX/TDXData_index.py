
import pandas as pd
import os
import sys

# Ensure current directory is in path to import TDXData
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from TDXData import download_all
except ImportError:
    print("Error: Could not import TDXData. Ensure TDXData.py is in the same directory.")
    sys.exit(1)

def main(dataset):
    map={
        "A500":"000510cons.xls",
        "A1000":"000852cons.xls",
        "HS300":"000300cons.xls",
        "HS500":"000905cons.xls",
        "AAll":"930903cons.xls",
        "KCCY50":"931643cons.xls",
        "KC50":"000688cons.xls",
    }
    # 1. Load Excel File
    excel_path = '/Volumes/IUSB/vipdoc/'+map[dataset]
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return

    print(f"Loading tickers from {excel_path}...")
    try:
        df_cons = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Setup Output Directory
    output_dir = '/Volumes/IUSB/vipdoc/TDX/'+dataset
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return
    else:
        print(f"Output directory exists: {output_dir}")

    # 3. Process Tickers
    success_count = 0
    fail_count = 0
    
    total_tickers = len(df_cons)
    print(f"Found {total_tickers} tickers to process.")

    for index, row in df_cons.iterrows():
        try:
            # Extract Ticker and Exchange
            raw_ticker = row['成份券代码Constituent Code']
            # Ensure ticker is 6 digits string
            ticker_code = f"{int(raw_ticker):06d}" 
            
            raw_exchange = row['交易所Exchange']
            
            # Map Exchange to SH/SZ
            if '上海' in str(raw_exchange) or 'Shanghai' in str(raw_exchange):
                exchange_suffix = 'sh'
            elif '深圳' in str(raw_exchange) or 'Shenzhen' in str(raw_exchange):
                exchange_suffix = 'sz'
            else:
                print(f"[{index+1}/{total_tickers}] Skipping {ticker_code}: Unknown exchange '{raw_exchange}'")
                fail_count += 1
                continue
            
            # Construct Symbol for TDXData (e.g. 000001.sz)
            symbol = f"{ticker_code}.{exchange_suffix}"
            # Download Data
            # print(f"[{index+1}/{total_tickers}] Processing {symbol}...", end='\r')
            
            df_data = download_all(symbol)
            
            if df_data is not None and not df_data.empty:
                # Save to CSV
                # Filename: Ticker Name (e.g. 000001.csv)
                # Adding exchange to filename to avoid collision if any (though unlikely for A-shares)
                # User requested "ticker name as filename". 
                # Interpret as "000001.csv" preferably, or "000001.sz.csv". 
                # Given strict instruction "using ticker name as filename", I will try to use just the code if unique.
                # But safer to just use code.
                
                out_filename = f"{ticker_code}.csv"
                out_path = os.path.join(output_dir, out_filename)
                
                df_data.to_csv(out_path)
                success_count += 1
            else:
                # print(f"[{index+1}/{total_tickers}] No data found for {symbol}")
                fail_count += 1
                
        except Exception as e:
            print(f"[{index+1}/{total_tickers}] Error processing row {index}: {e}")
            fail_count += 1
            
    print(f"\nProcessing Complete.")
    print(f"Successfully saved: {success_count}")
    print(f"Failed/No Data: {fail_count}")
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest WQ101 Alphas on Generic Dataset')
    parser.add_argument('--dataset', type=str, default='A500', help='Dataset folder name in TDX/ (e.g. A500, A1000)')
    parser.add_argument('--index_id', type=str, default=None, help='Index file ID (e.g. 000510, 000852). Optional.')
    args = parser.parse_args()
    
    DATASET_NAME = args.dataset

    main(DATASET_NAME)
