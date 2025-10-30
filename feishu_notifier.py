"""飞书消息通知模块"""
import json
import logging
from typing import List, Dict, Optional
import requests

logger = logging.getLogger(__name__)


class FeishuNotifier:
    """飞书 Webhook 通知器"""

    def __init__(self, webhook_url: str):
        """
        初始化飞书通知器
        
        Args:
            webhook_url: 飞书机器人的 Webhook URL
        """
        self.webhook_url = webhook_url

    def send_text(self, text: str) -> bool:
        """
        发送纯文本消息
        
        Args:
            text: 消息文本
            
        Returns:
            是否发送成功
        """
        payload = {
            "msg_type": "text",
            "content": {
                "name": text
            }
        }
        return self._send(payload)

    def send_stock_picks(
        self,
        alias: str,
        trade_date: str,
        picks: List[str],
        stock_names: Dict[str, str]
    ) -> bool:
        """
        发送选股结果通知
        
        Args:
            alias: 选股策略名称
            trade_date: 交易日期
            picks: 选中的股票代码列表
            stock_names: 股票代码到名称的映射
            
        Returns:
            是否发送成功
        """
        if not picks:
            content = f"📊 选股结果 [{alias}]\n\n"
            content += f"交易日: {trade_date}\n"
            content += f"符合条件股票数: 0\n"
            content += "无符合条件股票"
        else:
            # 格式化股票列表
            stock_list = []
            for code in picks:
                name = stock_names.get(code, "未知")
                stock_list.append(f"  • {code} ({name})")
            
            content = f"📊 选股结果 [{alias}]\n\n"
            content += f"交易日: {trade_date}\n"
            content += f"符合条件股票数: {len(picks)}\n\n"
            content += "选中股票:\n"
            content += "\n".join(stock_list)

        return self.send_text(content)

    def send_rich_text(
        self,
        title: str,
        alias: str,
        trade_date: str,
        picks: List[str],
        stock_names: Dict[str, str]
    ) -> bool:
        """
        发送富文本消息（支持更丰富的格式）
        
        Args:
            title: 消息标题
            alias: 选股策略名称
            trade_date: 交易日期
            picks: 选中的股票代码列表
            stock_names: 股票代码到名称的映射
            
        Returns:
            是否发送成功
        """
        # 构建富文本内容
        content = [[
            {"tag": "text", "text": f"策略: {alias}\n"},
            {"tag": "text", "text": f"交易日: {trade_date}\n"},
            {"tag": "text", "text": f"符合条件股票数: {len(picks)}\n\n"}
        ]]

        if picks:
            content[0].append({"tag": "text", "text": "选中股票:\n"})
            for code in picks:
                name = stock_names.get(code, "未知")
                content[0].append({
                    "tag": "text",
                    "text": f"  • {code} ({name})\n"
                })
        else:
            content[0].append({"tag": "text", "text": "无符合条件股票"})

        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": content
                    }
                }
            }
        }
        return self._send(payload)

    def _send(self, payload: dict) -> bool:
        """
        发送消息到飞书
        
        Args:
            payload: 消息载荷
            
        Returns:
            是否发送成功
        """
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.webhook_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            result = response.json()
            if result.get("code") == 0:
                logger.info("飞书消息发送成功")
                return True
            else:
                logger.error("飞书消息发送失败: %s", result.get("msg"))
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error("飞书消息发送异常: %s", e)
            return False
        except Exception as e:
            logger.error("飞书消息发送未知错误: %s", e)
            return False
