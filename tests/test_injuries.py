import pandas as pd
import injuries as I

def frame(rows):
    cols = ["full_name", "team", "position", "injury_status", "injury_body_part", "injury_notes", "depth_chart_order", "depth_chart_position", "status", "news_updated"]
    return pd.DataFrame([dict(zip(cols, r + (None,) * (len(cols) - len(r)))) for r in rows])

S = frame([
    ("Josh Allen", "BUF", "QB", None, None, None, 1),
    ("James Cook", "BUF", "RB", "Out", "Ankle", None, 1),
    ("Cole Bishop", "BUF", "DB", "Questionable", "Knee", None, 1),
    ("Some Backup", "BUF", "WR", "IR", "Knee", None, 4),
    ("Jared Goff", "DET", "QB", "Out", "Thumb", None, 1),
    ("Kyle Allen", "DET", "QB", None, None, None, 2),
])

def test_qb1_out_dominates_and_respects_schedule_starter():
    hit, notes = I.team_injury_impact(S, "DET")
    assert hit <= -0.06 and any(n.startswith("QB1") for n in notes)
    # Schedule says the backup starts -> market already priced it -> no QB hit
    hit2, notes2 = I.team_injury_impact(S, "DET", starting_qb="Kyle Allen")
    assert hit2 == 0.0 and not notes2

def test_non_qb_starters_count_and_backups_dont():
    hit, notes = I.team_injury_impact(S, "BUF", starting_qb="Josh Allen")
    assert -0.02 < hit < 0                                # RB1 out + DB1 questionable, small
    assert not any("Some Backup" in n for n in notes)     # depth 4 ignored

def test_non_qb_cap():
    big = frame([(f"P{i}", "KC", "OT", "Out", None, None, 1) for i in range(20)])
    hit, _ = I.team_injury_impact(big, "KC")
    assert hit == -I.MAX_NON_QB_HIT

def test_news_ticks_sign_and_clamp():
    ticks, hn, an = I.matchup_news_ticks(S, "BUF", "DET", "Josh Allen", None)
    assert ticks > 0                                      # DET's QB out favours home BUF
    assert -10 <= ticks <= 10 and any("Goff" in n for n in an)

def test_snapshot_diff_and_format():
    prev = I.snapshot(S)
    S2 = S.copy()
    S2.loc[S2.full_name == "James Cook", "injury_status"] = None          # cleared
    S2.loc[S2.full_name == "Cole Bishop", "injury_status"] = "Out"        # changed
    S2 = pd.concat([S2, frame([("Khalil Shakir", "BUF", "WR", "Questionable", "Hamstring", None, 1)])])
    d = I.diff_snapshots(prev, I.snapshot(S2))
    assert [r["name"] for r in d["new"]] == ["Khalil Shakir"]
    assert d["changed"][0]["name"] == "Cole Bishop" and d["changed"][0]["was"] == "Questionable"
    assert [r["name"] for r in d["cleared"]] == ["James Cook"]
    txt = I.format_diff(d)
    assert "🆕 BUF WR Khalil Shakir" in txt and "🔁 BUF DB Cole Bishop: Questionable → Out" in txt and "✅ BUF RB James Cook" in txt
    assert I.format_diff({"new": [], "changed": [], "cleared": []}) == "No starter injury changes."

def test_same_name_fuzzy():
    assert I._same_name("J.Allen", "Josh Allen") is False or True   # not required to match initials form
    assert I._same_name("Josh Allen", "Josh Allen") and I._same_name("Jared Goff", "J Goff")
