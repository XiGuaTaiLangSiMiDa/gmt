import os
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, CACHE_DIR

class DataFetcher:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.ensure_cache_dir()
        
    def ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def get_klines(self, interval, start_time=None):
        """获取K线数据"""
        cache_file = os.path.join(CACHE_DIR, f'{SYMBOL}_{interval}.csv')
        
        try:
            # 如果缓存文件存在，读取缓存
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 如果有缓存数据，获取最后一条数据的时间
                if not df.empty:
                    # 删除最后一条数据（因为它可能是不完整的）
                    last_complete_time = df.iloc[-2]['timestamp']
                    df = df.iloc[:-1]
                    
                    # 如果提供了start_time，使用较早的时间
                    if start_time:
                        start_time = min(start_time, last_complete_time)
                    else:
                        start_time = last_complete_time
            
            # 获取新数据
            klines = self.client.get_historical_klines(
                SYMBOL,
                interval,
                start_str=str(int(start_time.timestamp() * 1000)) if start_time else None
            )
            
            # 转换为DataFrame
            df_new = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 数据清洗和转换
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df_new[['open', 'high', 'low', 'close', 'volume']] = df_new[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # 合并新旧数据
            if os.path.exists(cache_file) and not df.empty:
                df = pd.concat([df, df_new])
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            else:
                df = df_new
                
            # 保存到缓存
            df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            print(f"获取数据时出错: {e}")
            # 如果有缓存数据，返回缓存数据
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            raise
    
    def get_all_timeframe_data(self):
        """获取所有时间周期的数据"""
        # GMT上线时间 2022-03-09
        start_time = datetime(2022, 3, 9)
        data = {}
        
        # 获取15分钟数据
        print("获取15分钟K线数据...")
        data['15m'] = self.get_klines('15m', start_time)
        
        # 获取其他时间周期数据
        intervals = ['30m', '1h', '2h', '4h', '8h', '12h', '1d', '3d', '1w']
        for interval in intervals:
            print(f"获取{interval}周期K线数据...")
            data[interval] = self.get_klines(interval, start_time)
            
        return data
