"""Write the README badges as self-contained SVG files (flat style, no external service).

    python scripts/make_badges.py                      # static badges -> docs/badges/
    python scripts/make_badges.py ci passed out.svg    # CI badge (used by the CI workflow)
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

# Approximate advance widths (px) of 11 px Verdana, the font shields-style badges use.
WIDTHS = {**dict.fromkeys("ijlI|.,:;'!", 3.5), **dict.fromkeys("frt()[]- ", 4.4),
          **dict.fromkeys("abcdeghknopqsuvxyz", 6.8), **dict.fromkeys("mw", 10.2),
          **dict.fromkeys("0123456789+", 7.0), **dict.fromkeys("ABCDEFGHJKLNOPQRSTUVXYZ", 7.6),
          "M": 8.6, "W": 10.8}

STATIC = {
    "preregistered": ("pre-registered", "2026-11-18", "#e8590c"),
    "python": ("python", "3.11+", "#2a78d6"),
    "licence": ("licence", "MIT", "#1baf7a"),
    "data": ("data", "WeatherBench 2", "#6250d6"),
}
CI_COLORS = {"passed": "#1baf7a", "failed": "#e34948"}


def text_width(s: str) -> float:
    return sum(WIDTHS.get(ch, 7.0) for ch in s)


def badge(label: str, message: str, color: str) -> str:
    lt, mt = text_width(label), text_width(message)
    lw, mw = round(lt + 12), round(mt + 12)
    w = lw + mw
    label_e, message_e = escape(label), escape(message)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" \
aria-label="{label_e}: {message_e}">
  <title>{label_e}: {message_e}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="20" fill="#555"/>
    <rect x="{lw}" width="{mw}" height="20" fill="{color}"/>
    <rect width="{w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" \
text-rendering="geometricPrecision" font-size="11">
{_text(lw / 2, lt, label_e)}{_text(lw + mw / 2, mt, message_e)}  </g>
</svg>
"""


def _text(x: float, length: float, content: str) -> str:
    """A text run with the one-pixel drop shadow used by flat badges."""
    shadow = 'fill="#010101" fill-opacity=".3"'
    return (f'    <text x="{x:.1f}" y="15" {shadow} textLength="{length:.1f}">{content}</text>\n'
            f'    <text x="{x:.1f}" y="14" textLength="{length:.1f}">{content}</text>\n')


def main(argv: list[str]) -> int:
    if argv[:1] == ["ci"]:
        status, out = argv[1], Path(argv[2])
        out.write_text(badge("CI", status, CI_COLORS.get(status, CI_COLORS["failed"])))
        return 0
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "badges"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in STATIC.items():
        (out_dir / f"{name}.svg").write_text(badge(*spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
