import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_fetcher import DataFetcher
from analyzer import MarketAnalyzer
from config import TIMEFRAMES

def create_visualization(data, timeframe):
    """创建交易数据可视化"""
    df = data[timeframe].copy()
    
    # 创建蜡烛图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # 添加K线图
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线'), row=1, col=1)

    # 添加成交量
    fig.add_trace(go.Bar(x=df['timestamp'],
                        y=df['volume'],
                        name='成交量'), row=2, col=1)

    # 更新布局
    fig.update_layout(
        title=f'GMT/USDT {timeframe} K线图',
        yaxis_title='价格',
        yaxis2_title='成交量',
        xaxis_rangeslider_visible=False
    )

    # 保存图表
    if not os.path.exists('reports'):
        os.makedirs('reports')
    fig.write_html(f'reports/chart_{timeframe}.html')

def format_pattern_report(patterns):
    """格式化交易模式报告"""
    report = []
    for timeframe, pattern in patterns.items():
        report.append(f"\n{timeframe} 时间周期分析:")
        report.append(f"- 偏好交易时间: {pattern['preferred_hours']}")
        report.append(f"- 平均价格影响: {pattern['avg_price_impact']:.2%}")
        report.append(f"- 平均成交量比率: {pattern['avg_volume_ratio']:.2f}")
        report.append(f"- 顺势交易比例: {pattern['trend_following_ratio']:.2%}")
        report.append(f"- 反转交易比例: {pattern['reversal_ratio']:.2%}")
    return '\n'.join(report)

def format_prediction_report(predictions):
    """格式化预测报告"""
    report = []
    for timeframe, pred in predictions.items():
        report.append(f"\n{timeframe} 预测结果:")
        report.append(f"- 预测趋势: {pred['trend']}")
        report.append(f"- 趋势概率: {pred['probability']:.2%}")
        report.append(f"- 预测价格: {pred['predicted_price']:.4f}")
        report.append(f"- 预测变化: {pred['predicted_change']:.2f}%")
        report.append(f"- 模型准确度: {pred['accuracy']:.2%}")
    return '\n'.join(report)

def format_strategy_report(strategies):
    """格式化策略报告"""
    report = []
    
    for term, strategy in strategies.items():
        report.append(f"\n## {term} 策略建议:")
        
        # 现货策略
        report.append("\n### 现货策略:")
        spot = strategy['spot']
        report.append(f"- 方向: {'做多' if spot['direction'] == 'up' else '做空'}")
        report.append(f"- 建议入场点: {[f'{price:.4f}' for price in spot['entry_points']]}")
        report.append(f"- 止损价位: {spot['stop_loss']:.4f}")
        report.append(f"- 止盈价位: {spot['take_profit']:.4f}")
        report.append(f"- 信心指数: {spot['confidence']:.2%}")
        
        # 合约策略
        report.append("\n### 合约策略:")
        futures = strategy['futures']
        report.append(f"- 方向: {'做多' if futures['direction'] == 'up' else '做空'}")
        report.append(f"- 建议杠杆: {futures['leverage']}倍")
        report.append(f"- 建议入场点: {[f'{price:.4f}' for price in futures['entry_points']]}")
        report.append(f"- 止损价位: {futures['stop_loss']:.4f}")
        report.append(f"- 止盈价位: {futures['take_profit']:.4f}")
        report.append(f"- 信心指数: {futures['confidence']:.2%}")
        
        # 风险提示
        if futures['confidence'] < 0.6:
            report.append("\n⚠️ 风险提示: 当前市场信号较弱，建议降低仓位或观望。")
        if futures['confidence'] < 0.4:
            report.append("⚠️ 高风险警告: 市场信号非常不明确，不建议进行合约交易。")
            
    return '\n'.join(report)

def main():
    # 创建报告目录
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # 获取数据
    print("正在获取数据...")
    fetcher = DataFetcher()
    data = fetcher.get_all_timeframe_data()

    # 分析数据
    print("正在分析数据...")
    analyzer = MarketAnalyzer(data)
    analysis = analyzer.get_analysis_report()

    # 生成报告
    print("正在生成报告...")
    report = []
    report.append("# GMT/USDT 市场分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## 操盘手风格分析")
    report.append(format_pattern_report(analysis['trading_patterns']))
    
    report.append("\n## 趋势预测")
    report.append(format_prediction_report(analysis['predictions']))
    
    report.append("\n# 交易策略建议")
    report.append(format_strategy_report(analysis['strategies']))
    
    # 保存报告
    with open('reports/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 生成可视化
    print("正在生成可视化...")
    for timeframe in data.keys():
        create_visualization(data, timeframe)

    print("分析完成！报告已保存到 reports 目录")

if __name__ == "__main__":
    main()
