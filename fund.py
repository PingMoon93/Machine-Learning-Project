#基金净值爬取脚本
import requests
import pandas as pd
import re

def get_fund_history(fund_code, start_date="2020-01-01", end_date="2025-08-01"):
    """
    爬取天天基金网基金净值历史
    :param fund_code: 基金代码，例如 "161725"
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :return: DataFrame [date, nav, acc_nav]
    """
    url = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    res = requests.get(url)
    res.encoding = "utf-8"
    text = res.text

    # 解析 JS 里的历史净值数据
    nav_pattern = re.compile(r"Data_netWorthTrend = (.*?);")
    match = nav_pattern.search(text)
    if not match:
        raise ValueError("未找到基金净值数据")
    data = eval(match.group(1))

    # 转换为 DataFrame
    records = []
    for item in data:
        date = pd.to_datetime(item["x"], unit="ms")
        if start_date <= str(date.date()) <= end_date:
            records.append([date.date(), item["y"], item["equityReturn"], item["unitMoney"]])

    df = pd.DataFrame(records, columns=["date", "nav", "daily_return", "distribution"])
    return df

if __name__ == "__main__":
    fund_code = "009049"  # 在这里换成你需要的基金代码
    df = get_fund_history(fund_code, start_date="2022-01-01", end_date="2025-08-01")
    df.to_csv(f"{fund_code}_nav.csv", index=False, encoding="utf-8-sig")
    print(f"已保存 {fund_code}_nav.csv")
    print(df.head())
