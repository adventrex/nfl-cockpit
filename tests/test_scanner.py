import pandas as pd
import pytest
import scanner as S

TM = {"Buffalo Bills": "BUF", "Detroit Lions": "DET"}

def payload(dk_home=1.45, fd_home=1.43, mgm_home=1.60, dk_away=2.85, fd_away=2.95, mgm_away=2.50):
    def book(key, h, a, hsp=-5.5, hsp_price=1.91, asp_price=1.91):
        return {"key": key, "markets": [
            {"key": "h2h", "outcomes": [{"name": "Buffalo Bills", "price": h}, {"name": "Detroit Lions", "price": a}]},
            {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "price": hsp_price, "point": hsp}, {"name": "Detroit Lions", "price": asp_price, "point": -hsp}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 54.5}, {"name": "Under", "price": 1.91, "point": 54.5}]}]}
    return [{"home_team": "Buffalo Bills", "away_team": "Detroit Lions", "commence_time": "2026-09-18T00:15:00Z",
             "bookmakers": [book("draftkings", dk_home, dk_away), book("fanduel", fd_home, fd_away), book("betmgm", mgm_home, mgm_away, hsp=-3.5)]}]

def test_parse_and_consensus_edge_flags_the_outlier_book():
    q = S.parse_payload(payload(), TM)
    assert set(q.market) == {"h2h", "spreads", "totals"} and q.opp_dec.notna().all()
    e = S.consensus_edges(q, min_books=3)
    top = e.iloc[0]
    assert top.market == "h2h" and top.side == "home" and top.book == "betmgm"   # 1.60 on BUF when others say ~1.44
    assert top.edge > S.EDGE_FLAG
    # the same book's away price is correspondingly bad
    bad = e[(e.book == "betmgm") & (e.side == "away") & (e.market == "h2h")].iloc[0]
    assert bad.edge < 0

def test_arb_detection():
    q = S.parse_payload(payload(mgm_home=1.60, fd_away=2.95), TM)
    a = S.arbs(q)
    h2h = a[a.market == "h2h"].iloc[0]
    assert h2h.hold < 0.01                     # 1/1.60 + 1/2.95 = 0.964 -> -3.6% hold = arb
    assert {"betmgm", "fanduel"} <= {h2h.leg1.split("@")[1], h2h.leg2.split("@")[1]}

def test_cross_book_middle():
    q = S.parse_payload(payload(), TM)
    m = S.middles(q)
    # BUF -3.5 at MGM, DET +5.5 at DK/FD -> middle if BUF wins by 4 or 5
    assert len(m) >= 1 and m.iloc[0].window == "3.5–5.5" and 0 < m.iloc[0].p_middle < 0.2

def test_empty_payload():
    q = S.parse_payload([], TM)
    assert q.empty and S.consensus_edges(q).empty and S.arbs(q).empty and S.middles(q).empty
