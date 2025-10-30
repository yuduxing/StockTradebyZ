# 每日股票数据获取与选股任务

这个项目包含一个自动化的每日任务，用于获取股票数据并执行选股策略。

## 功能

1. 每天下午6点自动运行
2. 执行以下操作：
   - 运行 `fetch_kline.py` 获取最新的股票K线数据
   - 运行 `select_stock.py` 执行选股策略
   - 将选股结果通过飞书机器人发送

## 使用方法

### 安装依赖

确保已安装Python 3.7+，然后安装所需依赖：

```bash
pip install schedule
```

### 运行方式

1. **直接运行**（测试用）：
   ```bash
   python run_daily_task.py
   ```

2. **后台运行**（推荐用于生产环境）：
   ```bash
   nohup python run_daily_task.py > daily_task.log 2>&1 &
   ```

3. **使用系统服务**（Linux系统）：

   创建服务文件 `/etc/systemd/system/stock_task.service`：
   ```ini
   [Unit]
   Description=Daily Stock Task
   After=network.target

   [Service]
   Type=simple
   User=your_username
   WorkingDirectory=/path/to/StockTradebyZ
   ExecStart=/usr/bin/python3 /path/to/StockTradebyZ/run_daily_task.py
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   然后启用并启动服务：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable stock_task
   sudo systemctl start stock_task
   ```

## 日志

- 日志会同时输出到控制台和 `daily_task.log` 文件
- 每次运行都会记录开始时间、结束时间和执行状态

## 注意事项

1. 确保 `stocklist.csv` 和 `configs.json` 文件存在且配置正确
2. 确保有足够的磁盘空间存储K线数据
3. 确保网络连接正常，能够访问股票数据源和飞书API
4. 如果修改了飞书Webhook地址，请更新 `FEISHU_WEBHOOK` 变量

## 手动触发

如果需要手动触发任务（例如在测试时），可以直接运行：

```bash
python run_daily_task.py
```

按 Ctrl+C 可以停止程序。
