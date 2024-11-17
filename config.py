import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Binance API配置
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# 交易对配置
SYMBOL = 'GMTUSDT'

# 时间周期配置(分钟)
TIMEFRAMES = {
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '8h': 480,
    '12h': 720,
    '1d': 1440,
    '3d': 4320,
    '1w': 10080
}

# 数据存储路径
DATA_DIR = 'data'
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
