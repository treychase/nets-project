"""
Next-season TS% projection dashboard.

Renders the Bayesian AR model's next-season projections (produced by
project.ipynb) as a self-contained, sortable/searchable HTML page.

Usage:
    python dashboard.py              # rebuild bayesian_ar_output/dashboard.html and open it
    python dashboard.py --no-open     # rebuild only, don't launch a browser
"""

import argparse
import html as html_lib
import webbrowser
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "bayesian_ar_output/next_season_projections.csv"
DEFAULT_HTML = "bayesian_ar_output/dashboard.html"


def build_dashboard_html(df: pd.DataFrame) -> str:
    """df needs PLAYER_NAME, TEAM, PROJECTED_AGE, PROJECTED_TS_PCT, HDI_LOW,
    HDI_HIGH, already sorted descending by PROJECTED_TS_PCT."""
    dom_min = max(0.0, df["HDI_LOW"].min() - 0.02)
    dom_max = min(1.0, df["HDI_HIGH"].max() + 0.02)
    dom_span = dom_max - dom_min

    rows_html = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        pct = row["PROJECTED_TS_PCT"] * 100
        low = row["HDI_LOW"] * 100
        high = row["HDI_HIGH"] * 100
        bar_pct = (row["PROJECTED_TS_PCT"] - dom_min) / dom_span * 100
        range_left = (row["HDI_LOW"] - dom_min) / dom_span * 100
        range_width = (row["HDI_HIGH"] - row["HDI_LOW"]) / dom_span * 100
        name = html_lib.escape(str(row["PLAYER_NAME"]))
        team = html_lib.escape(str(row["TEAM"]))
        rows_html.append(f"""
        <tr>
          <td class="rank">{rank}</td>
          <td class="name">{name}</td>
          <td class="team">{team}</td>
          <td class="age">{row['PROJECTED_AGE']:.0f}</td>
          <td class="bar-cell">
            <div class="bar-track">
              <div class="bar-range" style="left:{range_left:.2f}%;width:{range_width:.2f}%;"></div>
              <div class="bar-fill" style="width:{bar_pct:.2f}%;"></div>
            </div>
          </td>
          <td class="value">{pct:.1f}%<span class="hdi">[{low:.1f}&ndash;{high:.1f}]</span></td>
        </tr>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Next-Season TS% Projections</title>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #f7f8fa;
    --ink: #16181d;
    --ink-secondary: #55606e;
    --ink-muted: #8a94a3;
    --border: #e3e7ec;
    --accent: #2f6fed;
    --accent-range: rgba(47, 111, 237, 0.22);
    --track: #e9edf2;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14161a;
      --surface: #1c1f25;
      --ink: #eef1f5;
      --ink-secondary: #aab3c0;
      --ink-muted: #77808f;
      --border: #2b2f37;
      --accent: #6b9bff;
      --accent-range: rgba(107, 155, 255, 0.28);
      --track: #262a31;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 32px 20px 64px;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--ink-secondary); font-size: 0.9rem; margin: 0 0 20px; }}
  .caveat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--ink-secondary);
    margin-bottom: 20px;
    line-height: 1.5;
  }}
  .controls {{ margin-bottom: 12px; }}
  #search {{
    width: 100%;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9rem;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  thead th {{
    text-align: left;
    color: var(--ink-muted);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
  }}
  thead th:hover {{ color: var(--ink-secondary); }}
  tbody tr {{ border-bottom: 1px solid var(--border); }}
  tbody tr:hover {{ background: var(--surface); }}
  td {{ padding: 8px 10px; vertical-align: middle; }}
  td.rank {{ color: var(--ink-muted); width: 32px; }}
  td.name {{ font-weight: 500; white-space: nowrap; }}
  td.team {{ color: var(--ink-secondary); width: 56px; }}
  td.age {{ color: var(--ink-secondary); width: 48px; }}
  td.value {{ width: 130px; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  td.value .hdi {{ color: var(--ink-muted); font-size: 0.75rem; margin-left: 6px; }}
  td.bar-cell {{ width: 220px; }}
  .bar-track {{
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: var(--track);
    overflow: hidden;
  }}
  .bar-range {{
    position: absolute;
    top: 0; bottom: 0;
    background: var(--accent-range);
  }}
  .bar-fill {{
    position: absolute;
    top: 0; bottom: 0; left: 0;
    background: var(--accent);
    border-radius: 4px;
  }}
  .footnote {{ margin-top: 20px; color: var(--ink-muted); font-size: 0.75rem; }}
  #empty {{ display: none; color: var(--ink-muted); padding: 24px 0; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Next-Season TS% Projections</h1>
  <p class="subtitle">Bayesian Gaussian AR(3) hierarchical model &mdash; sorted by projected true shooting percentage</p>
  <div class="caveat">
    Projections use whichever of each player's three most recent seasons of
    TS% are actually available (fewer for early-career players), age
    advanced by one year, and usage rate held at its most recent observed
    value (next season's actual usage is unknown, so this is a steady-state
    assumption, not a usage forecast). Bars show the posterior mean on a
    {dom_min * 100:.0f}&ndash;{dom_max * 100:.0f}% scale; the shaded band is the 94% credible
    interval &mdash; wider bands usually mean less history behind that
    projection.
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Filter by player or team&hellip;">
  </div>
  <table id="proj-table">
    <thead>
      <tr>
        <th data-key="rank" data-type="num">#</th>
        <th data-key="name" data-type="str">Player</th>
        <th data-key="team" data-type="str">Team</th>
        <th data-key="age" data-type="num">Age</th>
        <th class="unsortable">Range</th>
        <th data-key="value" data-type="num">Projected TS%</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}
    </tbody>
  </table>
  <div id="empty">No players match that search.</div>
  <p class="footnote">Generated from the Bayesian AR(3) hierarchical model in project.ipynb.</p>
</div>
<script>
  const search = document.getElementById('search');
  const tbody = document.querySelector('#proj-table tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const empty = document.getElementById('empty');

  search.addEventListener('input', () => {{
    const q = search.value.trim().toLowerCase();
    let visible = 0;
    rows.forEach(r => {{
      const haystack = (r.querySelector('.name').textContent + ' ' + r.querySelector('.team').textContent).toLowerCase();
      const match = haystack.includes(q);
      r.style.display = match ? '' : 'none';
      if (match) visible++;
    }});
    empty.style.display = visible === 0 ? 'block' : 'none';
  }});

  let sortState = {{key: 'value', dir: -1}};
  document.querySelectorAll('#proj-table thead th').forEach((th, colIdx) => {{
    if (!th.dataset.key) return;
    th.addEventListener('click', () => {{
      const key = th.dataset.key;
      sortState.dir = sortState.key === key ? -sortState.dir : -1;
      sortState.key = key;
      const type = th.dataset.type;
      const sorted = rows.slice().sort((a, b) => {{
        const av = a.children[colIdx].textContent.trim();
        const bv = b.children[colIdx].textContent.trim();
        const an = type === 'num' ? parseFloat(av) : av.toLowerCase();
        const bn = type === 'num' ? parseFloat(bv) : bv.toLowerCase();
        if (an < bn) return -1 * sortState.dir;
        if (an > bn) return 1 * sortState.dir;
        return 0;
      }});
      sorted.forEach(r => tbody.appendChild(r));
    }});
  }});
</script>
</body>
</html>
"""


def build_dashboard(csv_path: str = DEFAULT_CSV, html_path: str = DEFAULT_HTML) -> Path:
    """Read the projections CSV, render the dashboard, and write it to disk."""
    df = (
        pd.read_csv(csv_path)
        .sort_values("PROJECTED_TS_PCT", ascending=False)
        .reset_index(drop=True)
    )
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_dashboard_html(df))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build (and open) the next-season TS% dashboard.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Projections CSV (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_HTML, help="Output HTML path (default: %(default)s)")
    parser.add_argument("--no-open", action="store_true", help="Don't launch a browser after building")
    args = parser.parse_args()

    out_path = build_dashboard(args.csv, args.output)
    print(f"Dashboard written to {out_path}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
