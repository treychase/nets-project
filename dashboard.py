"""
Next-season breakout dashboard.

Renders the combined Bayesian AR model projections (TS%, assist rate,
rebound rate, defensive rating -- produced by build_projections.py /
project.ipynb) as a self-contained, sortable/searchable HTML page, ranked by
each player's composite expected year-over-year improvement.

Usage:
    python dashboard.py              # rebuild bayesian_ar_output/dashboard.html and open it
    python dashboard.py --no-open     # rebuild only, don't launch a browser
"""

import argparse
import html as html_lib
import webbrowser
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "bayesian_ar_output/next_season_projections_all.csv"
DEFAULT_HTML = "bayesian_ar_output/dashboard.html"
DEFAULT_BOX_SCORE_CSV = "nba_data.csv"

# target column -> (label, unit suffix, decimals, higher values are better?)
STAT_COLUMNS = [
    ("TS_PCT", "TS%", "%", 1, True),
    ("AST_PCT", "AST%", "%", 1, True),
    ("REB_PCT", "REB%", "%", 1, True),
    ("DEF_RATING", "DEF RTG", "", 1, False),
]


def latest_fga_per_game(box_score_csv: str = DEFAULT_BOX_SCORE_CSV) -> pd.DataFrame:
    """Each player's FGA (already a per-game average) from their most recent season."""
    box = pd.read_csv(box_score_csv)
    latest = box.sort_values("SEASON").groupby("PLAYER_ID", as_index=False).last()
    return latest[["PLAYER_ID", "FGA"]].rename(columns={"FGA": "FGA_PER_GAME"})


def _fmt_stat_value(value: float, unit: str, decimals: int) -> str:
    if unit == "%":
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def _stat_cell_html(row: pd.Series, target_col: str, unit: str, decimals: int) -> str:
    current = row[f"CURRENT_{target_col}"]
    projected = row[f"PROJECTED_{target_col}"]
    delta = row[f"DELTA_{target_col}"]
    is_improvement = delta > 0  # DELTA_* is already direction-adjusted (positive = better)
    delta_class = "pos" if is_improvement else ("neg" if delta < 0 else "flat")
    raw_delta = projected - current  # shown in the stat's own units, not the direction-adjusted one
    sign = "+" if raw_delta >= 0 else ""
    if unit == "%":
        delta_str = f"{sign}{raw_delta * 100:.{decimals}f}"
    else:
        delta_str = f"{sign}{raw_delta:.{decimals}f}"
    return (
        f'<span class="stat-cur">{_fmt_stat_value(current, unit, decimals)}</span>'
        f'<span class="stat-arrow">&rarr;</span>'
        f'<span class="stat-proj">{_fmt_stat_value(projected, unit, decimals)}</span>'
        f'<span class="stat-delta {delta_class}">{delta_str}</span>'
    )


def build_dashboard_html(df: pd.DataFrame) -> str:
    """df needs PLAYER_NAME, TEAM, PROJECTED_AGE, FGA_PER_GAME, BREAKOUT_SCORE,
    and CURRENT_*/PROJECTED_*/DELTA_* for every column in STAT_COLUMNS,
    already sorted descending by BREAKOUT_SCORE."""
    score_abs_max = max(abs(df["BREAKOUT_SCORE"].min()), abs(df["BREAKOUT_SCORE"].max()), 0.01)

    teams = sorted(df["TEAM"].dropna().unique())
    team_options = "".join(
        f'<option value="{html_lib.escape(str(t))}">{html_lib.escape(str(t))}</option>' for t in teams
    )
    fga_min = float(df["FGA_PER_GAME"].min())
    fga_max = float(df["FGA_PER_GAME"].max())

    stat_headers = "".join(
        f'<th data-key="{col.lower()}" data-type="num">{label}</th>'
        for col, label, _unit, _dec, _hib in STAT_COLUMNS
    )

    rows_html = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        score = row["BREAKOUT_SCORE"]
        bar_pct = abs(score) / score_abs_max * 50  # each half of a centered track is 50%
        bar_side = "pos" if score >= 0 else "neg"
        name = html_lib.escape(str(row["PLAYER_NAME"]))
        team = html_lib.escape(str(row["TEAM"]))
        fga = row["FGA_PER_GAME"]

        stat_cells = "".join(
            f'<td class="stat-cell" data-{col.lower()}="{row[f"DELTA_{col}"]:.6f}">'
            f'{_stat_cell_html(row, col, unit, dec)}</td>'
            for col, _label, unit, dec, _hib in STAT_COLUMNS
        )

        rows_html.append(f"""
        <tr data-team="{team}" data-fga="{fga:.2f}">
          <td class="rank">{rank}</td>
          <td class="name">{name}</td>
          <td class="team">{team}</td>
          <td class="age">{row['PROJECTED_AGE']:.0f}</td>
          <td class="fga">{fga:.1f}</td>
          {stat_cells}
          <td class="bar-cell">
            <div class="bar-track">
              <div class="bar-mid"></div>
              <div class="bar-fill {bar_side}" style="width:{bar_pct:.2f}%;"></div>
            </div>
          </td>
          <td class="value">{score:+.2f}</td>
        </tr>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Next-Season Breakout Dashboard</title>
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
    --pos: #1a9c5c;
    --neg: #d34848;
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
      --pos: #3ecb85;
      --neg: #ef6a6a;
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
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
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
  .controls {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }}
  #search {{
    flex: 1 1 220px;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9rem;
  }}
  #team-filter {{
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9rem;
  }}
  .fga-filter {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--ink-secondary);
    font-size: 0.82rem;
    white-space: nowrap;
  }}
  .fga-filter input[type="range"] {{ accent-color: var(--accent); }}
  #fga-value {{ color: var(--ink); font-variant-numeric: tabular-nums; min-width: 3.2em; }}
  .table-scroll {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
  thead th {{
    text-align: left;
    color: var(--ink-muted);
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{ color: var(--ink-secondary); }}
  tbody tr {{ border-bottom: 1px solid var(--border); }}
  tbody tr:hover {{ background: var(--surface); }}
  td {{ padding: 7px 10px; vertical-align: middle; white-space: nowrap; }}
  td.rank {{ color: var(--ink-muted); width: 32px; }}
  td.name {{ font-weight: 500; }}
  td.team {{ color: var(--ink-secondary); width: 52px; }}
  td.age {{ color: var(--ink-secondary); width: 44px; }}
  td.fga {{ color: var(--ink-secondary); width: 52px; font-variant-numeric: tabular-nums; }}
  td.stat-cell {{ font-variant-numeric: tabular-nums; font-size: 0.8rem; color: var(--ink-secondary); }}
  .stat-arrow {{ margin: 0 3px; color: var(--ink-muted); }}
  .stat-proj {{ color: var(--ink); font-weight: 500; }}
  .stat-delta {{ margin-left: 6px; font-size: 0.72rem; font-weight: 600; }}
  .stat-delta.pos {{ color: var(--pos); }}
  .stat-delta.neg {{ color: var(--neg); }}
  .stat-delta.flat {{ color: var(--ink-muted); }}
  td.value {{ width: 64px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}
  td.bar-cell {{ width: 140px; }}
  .bar-track {{
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: var(--track);
    overflow: hidden;
  }}
  .bar-mid {{
    position: absolute;
    top: 0; bottom: 0; left: 50%;
    width: 1px;
    background: var(--border);
  }}
  .bar-fill {{
    position: absolute;
    top: 0; bottom: 0;
    border-radius: 4px;
  }}
  .bar-fill.pos {{ left: 50%; background: var(--pos); }}
  .bar-fill.neg {{ right: 50%; background: var(--neg); }}
  .footnote {{ margin-top: 20px; color: var(--ink-muted); font-size: 0.75rem; }}
  #empty {{ display: none; color: var(--ink-muted); padding: 24px 0; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Next-Season Breakout Dashboard</h1>
  <p class="subtitle">Bayesian flexible-lag AR(3) models &mdash; ranked by expected year-over-year improvement across shooting, playmaking, rebounding, and defense</p>
  <div class="caveat">
    Each of TS%, assist rate (AST%), rebound rate (REB%), and defensive
    rating is projected independently from age, usage rate, and up to three
    prior seasons of that stat (fewer for players with less history).
    <strong>Breakout score</strong> standardizes each stat's projected
    change (current &rarr; projected next season, flipped for defensive
    rating since a lower number is better there) and averages the four,
    so players are ranked by expected across-the-board improvement rather
    than by a single stat. Players already near the top of a stat have less
    room to climb and the model reverts them toward the population mean, so
    established stars often show a negative score here &mdash; this ranks
    who's expected to <em>improve</em>, not who will be best. Similarly, a
    very positive score for a low-minutes player often reflects limited
    history being pulled toward the mean rather than a real signal &mdash;
    check FGA/game before reading too much into it. Usage rate is held at
    its most recent observed value for all four models.
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Filter by player or team&hellip;">
    <select id="team-filter">
      <option value="">All teams</option>
      {team_options}
    </select>
    <div class="fga-filter">
      <label for="fga-min">FGA/game &ge;</label>
      <input id="fga-min" type="range" min="{fga_min:.1f}" max="{fga_max:.1f}" step="0.1" value="{fga_min:.1f}">
      <span id="fga-value">{fga_min:.1f}</span>
    </div>
  </div>
  <div class="table-scroll">
  <table id="proj-table">
    <thead>
      <tr>
        <th data-key="rank" data-type="num">#</th>
        <th data-key="name" data-type="str">Player</th>
        <th data-key="team" data-type="str">Team</th>
        <th data-key="age" data-type="num">Age</th>
        <th data-key="fga" data-type="num">FGA/G</th>
        {stat_headers}
        <th class="unsortable">Trend</th>
        <th data-key="value" data-type="num">Breakout</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}
    </tbody>
  </table>
  </div>
  <div id="empty">No players match that search.</div>
  <p class="footnote">Generated from the flexible-lag Bayesian AR(3) models in stat_models.py / build_projections.py.</p>
</div>
<script>
  const search = document.getElementById('search');
  const teamFilter = document.getElementById('team-filter');
  const fgaMin = document.getElementById('fga-min');
  const fgaValue = document.getElementById('fga-value');
  const tbody = document.querySelector('#proj-table tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const empty = document.getElementById('empty');

  function applyFilters() {{
    const q = search.value.trim().toLowerCase();
    const team = teamFilter.value;
    const minFga = parseFloat(fgaMin.value);
    let visible = 0;
    rows.forEach(r => {{
      const haystack = (r.querySelector('.name').textContent + ' ' + r.querySelector('.team').textContent).toLowerCase();
      const matchesSearch = haystack.includes(q);
      const matchesTeam = !team || r.dataset.team === team;
      const matchesFga = parseFloat(r.dataset.fga) >= minFga;
      const match = matchesSearch && matchesTeam && matchesFga;
      r.style.display = match ? '' : 'none';
      if (match) visible++;
    }});
    empty.style.display = visible === 0 ? 'block' : 'none';
  }}

  search.addEventListener('input', applyFilters);
  teamFilter.addEventListener('change', applyFilters);
  fgaMin.addEventListener('input', () => {{
    fgaValue.textContent = parseFloat(fgaMin.value).toFixed(1);
    applyFilters();
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
        const acell = a.children[colIdx];
        const bcell = b.children[colIdx];
        const attr = 'data-' + key;
        const av = acell.hasAttribute(attr) ? acell.getAttribute(attr) : acell.textContent.trim();
        const bv = bcell.hasAttribute(attr) ? bcell.getAttribute(attr) : bcell.textContent.trim();
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


def build_dashboard(
    csv_path: str = DEFAULT_CSV,
    html_path: str = DEFAULT_HTML,
    box_score_csv: str = DEFAULT_BOX_SCORE_CSV,
) -> Path:
    """Read the combined projections CSV, merge in each player's latest
    FGA/game, render the breakout dashboard, and write it to disk."""
    df = (
        pd.read_csv(csv_path)
        .merge(latest_fga_per_game(box_score_csv), on="PLAYER_ID", how="left")
        .sort_values("BREAKOUT_SCORE", ascending=False)
        .reset_index(drop=True)
    )
    df["FGA_PER_GAME"] = df["FGA_PER_GAME"].fillna(0.0)
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_dashboard_html(df))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build (and open) the next-season breakout dashboard.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Combined projections CSV (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_HTML, help="Output HTML path (default: %(default)s)")
    parser.add_argument(
        "--box-scores", default=DEFAULT_BOX_SCORE_CSV, help="Box-score CSV for FGA/game (default: %(default)s)"
    )
    parser.add_argument("--no-open", action="store_true", help="Don't launch a browser after building")
    args = parser.parse_args()

    out_path = build_dashboard(args.csv, args.output, args.box_scores)
    print(f"Dashboard written to {out_path}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
