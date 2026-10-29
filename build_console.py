"""Builds reports/capacity_console.html from reports/pipeline_output.json.
The page is a single self-contained file: open it in any browser."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent


def build_console() -> Path:
    data = json.loads((base / "reports" / "pipeline_output.json").read_text())
    for r in data["results"]:  # keep the page light: last 150 days of history
        r["history"] = {k: v[-150:] for k, v in r["history"].items()}
    html = (base / "console_template.html").read_text()
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = base / "reports" / "capacity_console.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    print("Wrote", build_console())
