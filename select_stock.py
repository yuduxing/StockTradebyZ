from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from feishu_notifier import FeishuNotifier

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def load_stock_names(stocklist_path: Path) -> Dict[str, str]:
    """加载股票代码到名称的映射"""
    if not stocklist_path.exists():
        logger.warning("股票列表文件 %s 不存在", stocklist_path)
        return {}
    try:
        # 确保 symbol 列作为字符串读取，保留前导零
        df = pd.read_csv(stocklist_path, dtype={"symbol": str})
        # 使用 symbol 作为 key (如 000001)，name 作为 value
        return dict(zip(df["symbol"], df["name"]))
    except Exception as e:
        logger.warning("加载股票名称失败: %s", e)
        return {}


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    p.add_argument("--feishu-webhook", help="飞书机器人 Webhook URL，用于发送通知")
    p.add_argument("--feishu-rich", action="store_true", help="使用富文本格式发送飞书消息")
    args = p.parse_args()

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载股票名称映射 ---
    stock_names = load_stock_names(data_dir.parent / "stocklist.csv")

    # --- 初始化飞书通知器 ---
    feishu = None
    if args.feishu_webhook:
        feishu = FeishuNotifier(args.feishu_webhook)
        logger.info("已启用飞书通知")

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        if picks:
            # 格式化输出：代码(名称)
            formatted_picks = [
                f"{code}({stock_names.get(code, '未知')})" for code in picks
            ]
            logger.info("%s", ", ".join(formatted_picks))
        else:
            logger.info("无符合条件股票")

        # 发送飞书通知
        if feishu:
            try:
                if args.feishu_rich:
                    feishu.send_rich_text(
                        title=f"选股结果 - {alias}",
                        alias=alias,
                        trade_date=str(trade_date.date()),
                        picks=picks,
                        stock_names=stock_names
                    )
                else:
                    feishu.send_stock_picks(
                        alias=alias,
                        trade_date=str(trade_date.date()),
                        picks=picks,
                        stock_names=stock_names
                    )
            except Exception as e:
                logger.error("发送飞书通知失败: %s", e)


if __name__ == "__main__":
    main()
