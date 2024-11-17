import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self, data):
        """
        初始化分析器
        data: 字典，键为时间周期，值为对应的DataFrame
        """
        self.data = data
        self.patterns = {}
        self.predictions = {}
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算波动率
        df['Volatility'] = df['close'].rolling(window=20).std()
        
        return df
        
    def analyze_trading_patterns(self):
        """分析交易模式和操盘手风格"""
        for timeframe, df in self.data.items():
            df = self.calculate_indicators(df.copy())
            
            # 分析大单交易
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            # 识别大单特征
            large_trades = df[
                (abs(df['price_change']) > df['price_change'].std() * 2) & 
                (df['volume_change'] > df['volume_change'].std() * 2)
            ]
            
            # 分析时间分布
            large_trades['hour'] = large_trades['timestamp'].dt.hour
            time_distribution = large_trades['hour'].value_counts()
            
            # 分析价格影响
            price_impact = large_trades.groupby(large_trades['timestamp'].dt.date)['price_change'].sum()
            
            # 统计操盘特征
            patterns = {
                'preferred_hours': time_distribution.nlargest(3).index.tolist(),
                'avg_price_impact': price_impact.mean(),
                'avg_volume_ratio': (large_trades['volume'] / df['volume'].mean()).mean(),
                'trend_following_ratio': (
                    (large_trades['price_change'] * large_trades['price_change'].shift(1) > 0).sum() / 
                    len(large_trades)
                ),
                'reversal_ratio': 1 - (
                    (large_trades['price_change'] * large_trades['price_change'].shift(1) > 0).sum() / 
                    len(large_trades)
                )
            }
            
            self.patterns[timeframe] = patterns
            
    def predict_trends(self):
        """预测不同时间维度的趋势"""
        for timeframe, df in self.data.items():
            df = self.calculate_indicators(df.copy())
            
            # 准备特征
            features = ['MA5', 'MA10', 'MA20', 'MACD', 'Signal', 'RSI', 'Volatility']
            X = df[features].dropna()
            y = df['close'].shift(-1).dropna()  # 预测下一个周期的收盘价
            
            # 确保数据对齐
            X = X.iloc[:-1]
            y = y.iloc[:len(X)]
            
            # 训练模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            train_size = int(len(X) * 0.8)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            model.fit(X_train, y_train)
            
            # 预测下一个和下两个周期
            last_data = X.iloc[-1:]
            next_period = model.predict(last_data)[0]
            
            # 计算预测准确度
            accuracy = model.score(X_test, y_test)
            
            # 计算趋势概率
            current_price = df['close'].iloc[-1]
            trend = 'up' if next_period > current_price else 'down'
            probability = abs(next_period - current_price) / current_price
            
            self.predictions[timeframe] = {
                'trend': trend,
                'probability': probability,
                'predicted_price': next_period,
                'accuracy': accuracy,
                'predicted_change': (next_period - current_price) / current_price * 100
            }
            
    def get_analysis_report(self):
        """生成分析报告"""
        self.analyze_trading_patterns()
        self.predict_trends()
        
        report = {
            'trading_patterns': self.patterns,
            'predictions': self.predictions
        }
        
        return report
