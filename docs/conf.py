import json
from pathlib import Path

release = json.loads(
    (Path(__file__).resolve().parents[1] / "package.json").read_text(encoding="utf-8")
)["version"]

project = "ONNX Light Viewer"
today_fmt = "%Y-%m-%d"
html_theme = "pydata_sphinx_theme"
html_logo = "../media/logo.svg"
html_favicon = "../media/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/onnx-light-viewer",
    "header_links_before_dropdown": 10,
}
