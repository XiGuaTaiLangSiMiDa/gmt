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
        self.data = {}
        # 预先计算所有指标
        for timeframe, df in data.items():
            self.data[timeframe] = self.calculate_indicators(df.copy())
        self.patterns = {}
        self.predictions = {}
        self.strategies = {}
        
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
        df['Volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # 计算布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + (df['close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['BB_middle'] - (df['close'].rolling(window=20).std() * 2)

        # 计算价格和成交量变化
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        return df
        
    def analyze_trading_patterns(self):
        """分析交易模式和操盘手风格"""
        for timeframe, df in self.data.items():
            # 识别大单特征
            large_trades = df[
                (abs(df['price_change']) > df['price_change'].std() * 2) & 
                (df['volume_change'] > df['volume_change'].std() * 2)
            ]
            
            # 分析时间分布
            large_trades.loc[:, 'hour'] = large_trades['timestamp'].dt.hour
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
                    len(large_trades) if len(large_trades) > 0 else 0
                ),
                'reversal_ratio': 1 - (
                    (large_trades['price_change'] * large_trades['price_change'].shift(1) > 0).sum() / 
                    len(large_trades) if len(large_trades) > 0 else 0
                )
            }
            
            self.patterns[timeframe] = patterns
            
    def predict_trends(self):
        """预测不同时间维度的趋势"""
        for timeframe, df in self.data.items():
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

    def generate_strategies(self):
        """生成交易策略建议"""
        self.strategies = {
            'short_term': self._generate_strategy('short'),
            'medium_term': self._generate_strategy('medium'),
            'long_term': self._generate_strategy('long')
        }

    def _generate_strategy(self, term):
        """生成特定时间维度的策略"""
        if term == 'short':
            timeframes = ['15m', '30m', '1h']
        elif term == 'medium':
            timeframes = ['4h', '8h', '12h']
        else:  # long
            timeframes = ['1d', '3d', '1w']

        strategy = {
            'spot': {
                'direction': None,
                'entry_points': [],
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0
            },
            'futures': {
                'direction': None,
                'leverage': None,
                'entry_points': [],
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0
            }
        }

        # 统计各时间周期的趋势
        trends = []
        for tf in timeframes:
            if tf in self.predictions:
                pred = self.predictions[tf]
                trends.append({
                    'trend': pred['trend'],
                    'probability': pred['probability'],
                    'accuracy': pred['accuracy']
                })

        if not trends:
            return strategy

        # 计算综合趋势
        up_probability = sum(1 for t in trends if t['trend'] == 'up') / len(trends)
        avg_probability = sum(t['probability'] for t in trends) / len(trends)
        avg_accuracy = sum(t['accuracy'] for t in trends) / len(trends)

        # 确定方向和信心度
        direction = 'up' if up_probability > 0.5 else 'down'
        confidence = (avg_probability + avg_accuracy) / 2

        # 获取当前价格和波动率
        current_price = self.data[timeframes[0]]['close'].iloc[-1]
        volatility = self.data[timeframes[0]]['Volatility'].iloc[-1]

        # 现货策略
        strategy['spot']['direction'] = direction
        strategy['spot']['confidence'] = confidence
        if direction == 'up':
            strategy['spot']['entry_points'] = [
                current_price * 0.99,  # 回调1%入场
                current_price * 0.98,  # 回调2%入场
                current_price * 0.97   # 回调3%入场
            ]
            strategy['spot']['stop_loss'] = current_price * 0.95  # 5%止损
            strategy['spot']['take_profit'] = current_price * 1.1  # 10%获利
        else:
            strategy['spot']['entry_points'] = [
                current_price,
                current_price * 1.01,  # 上涨1%卖出
                current_price * 1.02   # 上涨2%卖出
            ]
            strategy['spot']['stop_loss'] = current_price * 1.05  # 5%止损
            strategy['spot']['take_profit'] = current_price * 0.9  # 10%获利

        # 合约策略
        strategy['futures']['direction'] = direction
        strategy['futures']['confidence'] = confidence
        
        # 根据信心度和波动率确定杠杆
        base_leverage = 3 if confidence > 0.7 else (2 if confidence > 0.5 else 1)
        vol_adj_leverage = base_leverage * (1 - volatility)  # 波动率越大，杠杆越小
        strategy['futures']['leverage'] = max(1, min(5, round(vol_adj_leverage)))  # 限制杠杆在1-5倍

        if direction == 'up':
            strategy['futures']['entry_points'] = [
                current_price * 0.995,  # 回调0.5%入场
                current_price * 0.99,   # 回调1%入场
                current_price * 0.985   # 回调1.5%入场
            ]
            strategy['futures']['stop_loss'] = current_price * 0.97  # 3%止损
            strategy['futures']['take_profit'] = current_price * 1.06  # 6%获利
        else:
            strategy['futures']['entry_points'] = [
                current_price,
                current_price * 1.005,  # 上涨0.5%做空
                current_price * 1.01    # 上涨1%做空
            ]
            strategy['futures']['stop_loss'] = current_price * 1.03  # 3%止损
            strategy['futures']['take_profit'] = current_price * 0.94  # 6%获利

        return strategy
            
    def get_analysis_report(self):
        """生成分析报告"""
        self.analyze_trading_patterns()
        self.predict_trends()
        self.generate_strategies()
        
        report = {
            'trading_patterns': self.patterns,
            'predictions': self.predictions,
            'strategies': self.strategies
        }
        
        return report
