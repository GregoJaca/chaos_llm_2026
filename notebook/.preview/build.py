"""Build notebook.md -> notebook.html (KaTeX math, offline) -> notebook.pdf (Edge/Chrome headless).

Usage:
  python .preview/build.py            # build once
  python .preview/build.py --watch    # rebuild on change
  python .preview/build.py --serve    # build, serve over HTTP (Tailscale), watch
"""

import base64
import html as htmllib
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(NOTE_DIR)          # chaos_llm_2026
SRC = os.path.join(NOTE_DIR, "notebook.md")
HTML_OUT = os.path.join(NOTE_DIR, "notebook.html")
PDF_OUT = os.path.join(NOTE_DIR, "notebook.pdf")
KATEX = os.path.join(HERE, "katex")
PORT = int(os.environ.get("NOTEBOOK_PORT", "8899"))

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#d8d8d8; --accent:#0b5cad; }
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Cambria", serif;
  font-size: 15px; line-height: 1.55; color: var(--fg);
  margin: 0 auto; max-width: 210mm; padding: 18mm 16mm 24mm;
  background: #fff;
}
h1, h2, h3, h4 { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; line-height:1.25; }
h1 { font-size: 1.65em; margin: 1.4em 0 .5em; border-bottom: 2px solid var(--line); padding-bottom:.25em; }
h2 { font-size: 1.3em; margin: 1.3em 0 .4em; color:#111; }
h3 { font-size: 1.08em; margin: 1.1em 0 .3em; color:#222; }
p, li { margin: .45em 0; }
ul, ol { margin: .45em 0 .45em 1.3em; padding: 0; }
code, pre { font-family: "Cascadia Mono", Consolas, monospace; font-size: .86em; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }
pre { background:#f7f7f7; border:1px solid #e5e5e5; border-radius:4px; padding:8px 10px; overflow-x:auto; }
img { max-width: 100%; height: auto; display:block; margin: 10px auto; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: .93em; }
th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f5f7f9; }
hr { border: 0; border-top: 1px solid var(--line); margin: 1.8em 0; }
blockquote { margin: .6em 0; padding: .3em 1em; border-left: 3px solid var(--line); color:#333; }
a { color: var(--accent); }
.katex-display { margin: .7em 0; overflow-x: auto; overflow-y: hidden; }
.katex { font-size: 1.03em; }
header.banner {
  font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; color: var(--muted);
  border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 18px;
}
header.banner a { text-decoration: none; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
img, table, pre, .katex-display { break-inside: avoid; page-break-inside: avoid; }
@page { size: A4; margin: 15mm 14mm 16mm; }
"""


def protect_math(text):
    """Replace $...$ / $$...$$ with tokens so markdown doesn't mangle them."""
    store = []

    def repl(m):
        store.append(m.group(0))
        return "\u2063MATH%d\u2063" % (len(store) - 1)

    text = re.sub(r"\$\$.+?\$\$", repl, text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\$)[^\n$]+?\$(?!\$)", repl, text)
    return text, store


def restore_math(html, store):
    def repl(m):
        return store[int(m.group(1))]

    return re.sub("\u2063MATH(\\d+)\u2063", repl, html)


def build():
    import markdown

    t0 = time.time()
    with open(SRC, "r", encoding="utf-8") as f:
        md = f.read()

    body_md, math = protect_math(md)
    body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
    )
    body = restore_math(body, math)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>notebook.md</title>
<link rel="stylesheet" href=".preview/katex/katex.min.css">
<style>{CSS}</style>
</head><body>
<header class="banner">
  built {stamp} &nbsp;·&nbsp; source: notebook.md &nbsp;·&nbsp;
  <a href="notebook.pdf">PDF</a> &nbsp;·&nbsp; <a href="notebook.md">Markdown</a>
</header>
{body}
<script src=".preview/katex/katex.min.js"></script>
<script src=".preview/katex/contrib/auto-render.min.js"></script>
<script>
renderMathInElement(document.body, {{
  delimiters: [{{left:"$$", right:"$$", display:true}}, {{left:"$", right:"$", display:false}}],
  throwOnError: false,
  strict: false
}});
</script>
</body></html>
"""
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(page)

    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        print("! no Edge/Chrome found; wrote HTML only")
        return False

    tmp_pdf = PDF_OUT + ".tmp.pdf"
    if os.path.exists(tmp_pdf):
        os.remove(tmp_pdf)
    cmd = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={tmp_pdf}",
        "file:///" + HTML_OUT.replace("\\", "/"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 10000
    if ok:
        os.replace(tmp_pdf, PDF_OUT)
    else:
        print("! pdf failed:", (r.stderr or r.stdout)[-600:])
    print(
        "built %s  html %.0f kB  pdf %.0f kB  in %.1fs"
        % (stamp, os.path.getsize(HTML_OUT) / 1e3,
           os.path.getsize(PDF_OUT) / 1e3 if os.path.exists(PDF_OUT) else 0,
           time.time() - t0)
    )
    return ok


def local_ips():
    ips = set()
    try:
        for line in subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split():
            if line.strip():
                ips.add(line.strip())
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips)


def serve():
    os.chdir(NOTE_DIR)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("serving %s on port %d" % (NOTE_DIR, PORT))
    for ip in local_ips():
        print("  http://%s:%d/notebook.html" % (ip, PORT))
        print("  http://%s:%d/notebook.pdf" % (ip, PORT))


def watch():
    last = None
    pending = False
    changed_at = 0.0
    print("watching", SRC)
    while True:
        try:
            m = os.path.getmtime(SRC)
        except OSError:
            time.sleep(1)
            continue
        if last is None:
            last = m
        elif m != last:
            last = m
            pending = True
            changed_at = time.time()
        if pending and time.time() - changed_at > 1.2:
            pending = False
            try:
                build()
                print("  (re)built after change", time.strftime("%H:%M:%S"))
            except Exception as e:
                print("! build error:", e)
        time.sleep(0.5)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build()
        serve()
        watch()
    elif "--watch" in sys.argv:
        build()
        watch()
    else:
        sys.exit(0 if build() else 1)
