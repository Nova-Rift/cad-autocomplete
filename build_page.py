"""Write web/index.html (the stand-alone page) from web/body.html, the one copy of the page's content.

    python build_page.py
"""
from pathlib import Path
web = Path(__file__).parent / "web"
LEAD = "Draw a shape. The blue dashed shape is the model's guess at what you'll draw next. Press <b>Tab</b> to take it."
web.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sketch autocomplete</title>
<link rel="stylesheet" href="sketch.css">
</head>
<body class="sk-alone">
<main>
<h1>Sketch autocomplete</h1>
<p class="sk-lead">{LEAD}</p>
{web.joinpath("body.html").read_text()}</main>
<script type="module" src="app.js"></script>
</body>
</html>
""")
print("wrote web/index.html")
