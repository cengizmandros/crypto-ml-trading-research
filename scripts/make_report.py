"""Render reports/report.html from the research (and, if run, holdout) results."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from traderbot.config import REPORTS

NAMES = {"btc_buy_hold": "BTC al-ve-tut", "ew_universe_monthly": "Eşit ağırlık evren (aylık)",
         "btc_trend_ma200": "BTC trend (MA200)"}


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def table(results: dict, selected: str) -> str:
    rows = sorted(results.items(), key=lambda kv: -(kv[1].get("sharpe") or -9))
    h = ["<table><tr><th>Strateji</th><th>Toplam</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th>"
         "<th>Calmar</th><th>Kazanma</th><th>İşlem</th><th>Ort. pozisyon</th><th>Maliyet / başl. sermaye</th></tr>"]
    for k, v in rows:
        cls = ' class="sel"' if k == selected else ""
        h.append(f"<tr{cls}><td>{NAMES.get(k, k)}</td><td>{pct(v.get('total_return'))}</td><td>{pct(v.get('cagr'))}</td>"
                 f"<td>{num(v.get('sharpe'))}</td><td>{num(v.get('sortino'))}</td><td>{pct(v.get('max_dd'))}</td>"
                 f"<td>{num(v.get('calmar'))}</td><td>{pct(v.get('win_rate'), 0)}</td><td>{v.get('n_trades', '–')}</td>"
                 f"<td>{pct(v.get('exposure'), 0)}</td><td>{pct(v.get('costs_paid_pct_of_start'))}</td></tr>")
    h.append("</table>")
    return "".join(h)


_JS_DONE = False


def curves(path, keys) -> str:
    """First figure embeds plotly.js inline so the report works offline."""
    global _JS_DONE
    r = pd.read_parquet(path)
    fig = go.Figure()
    for k in keys:
        if k in r.columns:
            eq = (1 + r[k].fillna(0)).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=NAMES.get(k, k), mode="lines"))
    fig.update_layout(yaxis_type="log", height=420, margin=dict(l=40, r=10, t=10, b=30),
                      legend=dict(orientation="h", y=1.08), template="plotly_white")
    out = fig.to_html(full_html=False, include_plotlyjs=not _JS_DONE)
    _JS_DONE = True
    return out


def main():
    w = json.loads((REPORTS / "wf_results.json").read_text())
    sel = w["selected"]
    mode = sel.split("|")[1]
    keys = ["btc_buy_hold", "ew_universe_monthly", "btc_trend_ma200", sel, f"xs_momentum|{mode}"]
    rb = w["robustness"]
    ic_rows = "".join(f"<tr><td>{k}</td><td>{v['mean_ic']:.4f}</td><td>{v['ic_tstat']:.2f}</td><td>{pct(v['pct_days_pos'], 0)}</td></tr>"
                      for k, v in w["ic"].items())
    folds = "".join(f"<tr><td>{f['test_start']}</td><td>{f['n_train']}</td>" +
                    "".join(f"<td>{f['val_ic'].get(m, float('nan')):.4f}</td>" for m in ("lgbm", "xgb", "transformer")) + "</tr>"
                    for f in w["folds"])
    holdout = ""
    if (REPORTS / "holdout_results.json").exists():
        h = json.loads((REPORTS / "holdout_results.json").read_text())
        holdout = (f"<h2>Nihai test ({h['period'][0]} → {h['period'][1]}), tek seferlik</h2>"
                   + curves(REPORTS / "holdout_equity.parquet", keys) + table(h["results"], sel)
                   + f"<pre>{json.dumps({k: v for k, v in h['robustness'].items() if k != 'random_selection'}, indent=1, default=str)}</pre>"
                   + f"<p>Rastgele seçim testi: p = {h['robustness']['random_selection']['p_value']:.3f}</p>")
    rs = rb["random_selection"]
    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trader Bot Raporu</title><style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1150px;margin:auto;padding:16px;color:#1d1d1f;background:#fafaf8}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px;margin:8px 0 20px}}
th,td{{padding:4px 6px;border-bottom:1px solid #e3e3df;text-align:right}}th:first-child,td:first-child{{text-align:left}}
tr.sel td{{font-weight:700;background:#eef3ff}}pre{{background:#f0f0ec;padding:10px;overflow:auto;font-size:12px}}h2{{margin-top:32px}}</style></head><body>
<h1>Trader Bot — Araştırma Raporu</h1>
<p>Walk-forward örneklem dışı dönem: <b>{w['period'][0]} → {w['period'][1]}</b>. Hedef ufku {w['horizon']} gün.
{w['n_features']} özellik. Seçilen sistem (nihai test için önceden kilitlendi): <b>{sel}</b>.
Tüm sonuçlar %0,10 komisyon + likiditeye göre spread + slippage + piyasa etkisi sonrası.</p>
<h2>Bakiye eğrileri (log ölçek)</h2>{curves(REPORTS / 'wf_equity.parquet', keys)}
<h2>Tüm varyantlar</h2>{table(w['results'], sel)}
<h2>Sinyal kalitesi (örneklem dışı rank IC)</h2><table><tr><th>Sinyal</th><th>Ort. IC</th><th>t-stat (örtüşme düzeltmeli)</th><th>IC&gt;0 gün</th></tr>{ic_rows}</table>
<h2>Sağlamlık testleri (seçilen sistem)</h2>
<p>Rastgele coin seçimi (aynı risk katmanı, {rs['n']} simülasyon): ortalama Sharpe {rs['random_sharpe_mean']:.2f},
%95'lik {rs['random_sharpe_p95']:.2f}; p-değeri <b>{rs['p_value']:.3f}</b>.</p>
<pre>{json.dumps({k: v for k, v in rb.items() if k != 'random_selection'}, indent=1, default=str)}</pre>
<h2>Fold başına iç doğrulama IC</h2><table><tr><th>Test başlangıcı</th><th>Eğitim satırı</th><th>LightGBM</th><th>XGBoost</th><th>Transformer</th></tr>{folds}</table>
<p>Optuna deneme sayıları: {w['trials']}; örneklem dışında karşılaştırılan varyant sayısı: {w['n_variants']}.</p>
{holdout}
</body></html>"""
    (REPORTS / "report.html").write_text(html)
    print(REPORTS / "report.html")


if __name__ == "__main__":
    main()
