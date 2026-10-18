from app.market.binance_stream import summarize_trades


def test_agg_trade_aggressor_side_and_delta():
    trades = [
        {"p": "100", "q": "1000", "m": False, "T": 1000},
        {"p": "101", "q": "500", "m": True, "T": 2000},
    ]
    flow = summarize_trades(trades)
    assert flow.buy_quote == 100_000
    assert flow.sell_quote == 50_500
    assert flow.cvd_quote == 49_500
    assert flow.count == 2
    assert {item["side"] for item in flow.large_trades} == {"buy", "sell"}
