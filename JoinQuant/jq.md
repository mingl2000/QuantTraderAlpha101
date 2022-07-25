# Joinquant Api

为了熟悉 joinquant 框架，特意手写、手抄一遍这个框架。简单但是完整的策略

~~~python
def initialize(context):
    g.security = "000001.XSHE"
    run_daily(market_open, time='every_bar')

def market_open(context):
    if g.security not in context.protfolio.positions:
        order(g.security, 1000)
    else:
        order(g.security, -800)
~~~

一个完整的策略只需要两步：
1. 设置初始化函数 initialize,