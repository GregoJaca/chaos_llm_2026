"""Build markdown docs -> HTML (offline KaTeX) -> PDF (headless browser), serve over HTTP
(Tailscale) and rebuild automatically when the sources change.

Documents:
  notebook.md -> notebook.html / notebook.pdf
  master.md   -> master.html   / master.pdf

Usage:
  python .preview/build.py                # build everything once
  python .preview/build.py --watch        # rebuild on change
  python .preview/build.py --serve        # build, serve, keep watching

Self-healing: the loop rebuilds whenever a source is newer than its PDF, so a missed event,
a crashed browser, a killed process or a laptop restart all recover by themselves.
Every run is logged to .preview/build.log.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_DIR = os.path.dirname(HERE)
LOG_PATH = os.path.join(HERE, "build.log")
KATEX = os.path.join(HERE, "katex")
PORT = int(os.environ.get("NOTEBOOK_PORT", "8899"))

POLL_SECONDS = 2.0
RETRY_SECONDS = 20.0

# name, source file, short description shown on the index page
DOCS = [
    ("update", "update.md", "Update 2 &mdash; all old figures rebuilt from raw data, plus new analyses"),
    ("unified", "unified.md", "Unified treatise &mdash; figures regenerated from raw data; text is the old draft"),
    ("controls", "controls.md", "Controls &mdash; Mamba / random-transformer vs the LLM results"),
    ("master", "master.md", "Claims only &mdash; the results I would defend, with definitions"),
    ("veto", "veto.md", "Extra results and whether to include them (verdicts)"),
    ("new", "new.md", "Temperature bridge &mdash; the soft-generation loop, margin gates, sampling"),
    ("notebook", "notebook.md", "Lab notebook &mdash; everything, including dead ends"),
]

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
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: .93em; table-layout: fixed; }
th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top;
         word-break: break-word; overflow-wrap: anywhere; }
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


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 2_000_000:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def paths(name):
    return (os.path.join(NOTE_DIR, name + ".md"),
            os.path.join(NOTE_DIR, name + ".html"),
            os.path.join(NOTE_DIR, name + ".pdf"))


def is_stale(name):
    src, html, pdf = paths(name)
    if not os.path.exists(src):
        return False
    return mtime(src) > mtime(pdf) or mtime(src) > mtime(html)


def protect_math(text):
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


def build(name):
    """Render one document. Returns True on success, never raises."""
    import markdown

    src, html_out, pdf_out = paths(name)
    t0 = time.time()
    try:
        with open(src, "r", encoding="utf-8") as f:
            md = f.read()
    except Exception as e:
        log("! cannot read %s: %s" % (src, e))
        return False

    try:
        body_md, math = protect_math(md)
        body = markdown.markdown(
            body_md,
            extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
        )
        body = restore_math(body, math)
    except Exception:
        log("! markdown error in %s:\n%s" % (src, traceback.format_exc()))
        return False

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    others = " &middot; ".join(
        '<a href="%s.html">%s</a>' % (n, n) for n, _, _ in DOCS if n != name
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}.md</title>
<link rel="stylesheet" href=".preview/katex/katex.min.css">
<style>{CSS}</style>
</head><body>
<header class="banner">
  {name}.md &nbsp;&middot;&nbsp; built {stamp} &nbsp;&middot;&nbsp;
  <a href="{name}.pdf">PDF</a> &nbsp;&middot;&nbsp;
  <a href="{name}.md">Markdown</a> &nbsp;&middot;&nbsp;
  <a href="./">all documents</a> &nbsp;&middot;&nbsp; {others}
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
    try:
        tmp_html = html_out + ".tmp"
        with open(tmp_html, "w", encoding="utf-8") as f:
            f.write(page)
        os.replace(tmp_html, html_out)
    except Exception:
        log("! html write error:\n" + traceback.format_exc())
        return False

    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        log("! no Edge/Chrome found; wrote HTML only")
        return False

    tmp_pdf = pdf_out + ".tmp.pdf"
    try:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
    except OSError:
        pass

    cmd = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        "--virtual-time-budget=30000",
        f"--print-to-pdf={tmp_pdf}",
        "file:///" + html_out.replace("\\", "/"),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
    except subprocess.TimeoutExpired:
        log("! browser timed out after 420s for %s" % name)
        return False
    except Exception:
        log("! browser launch failed:\n" + traceback.format_exc())
        return False

    if os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 10_000:
        try:
            os.replace(tmp_pdf, pdf_out)
        except Exception:
            log("! cannot replace pdf: %s" % traceback.format_exc())
            return False
    else:
        tail = (r.stderr or r.stdout or "")[-600:]
        log("! pdf failed for %s (exit %s): %s" % (name, r.returncode, tail.replace("\n", " ")))
        try:
            os.remove(tmp_pdf)
        except OSError:
            pass
        return False

    log("built %s -> %s.html %.0f kB, %s.pdf %.1f MB in %.1fs"
        % (name, name, os.path.getsize(html_out) / 1e3, name,
           os.path.getsize(pdf_out) / 1e6, time.time() - t0))
    return True


def build_all():
    ok = True
    for name, _, _ in DOCS:
        if not os.path.exists(paths(name)[0]):
            continue
        ok = build(name) and ok
    return ok


INDEX = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>documents</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;margin:0;padding:28px 20px;
max-width:680px;margin:0 auto;color:#111;background:#fafafa}
h1{font-size:1.35em;margin:0 0 4px}
p.meta{color:#666;font-size:.85em;margin:0 0 26px}
h2{font-size:.95em;color:#666;font-weight:600;margin:26px 0 8px;text-transform:uppercase;
letter-spacing:.06em}
a.card{display:block;text-decoration:none;color:inherit;background:#fff;border:1px solid #ddd;
border-radius:10px;padding:16px 18px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
a.card b{display:block;font-size:1.05em;margin-bottom:3px}
a.card span{color:#666;font-size:.85em}
</style></head><body>
<h1>documents</h1>
<p class="meta">served over Tailscale &middot; rebuilt automatically when the .md changes</p>
%s
</body></html>
"""


def index_html():
    cards = []
    for name, _, desc in DOCS:
        src, html, pdf = paths(name)
        if not os.path.exists(src):
            continue
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime(src)))
        pdf_mb = os.path.getsize(pdf) / 1e6 if os.path.exists(pdf) else None
        size = (" &middot; %.1f MB PDF" % pdf_mb) if pdf_mb else ""
        cards.append(
            '<a class="card" href="%s.html"><b>%s</b><span>%s &middot; edited %s%s '
            '&nbsp;|&nbsp; <a href="%s.pdf">PDF</a></span></a>' % (name, name, desc, stamp, size, name)
        )
    return INDEX % ("\n".join(cards) or "<p>no documents yet</p>")


def serve():
    os.chdir(NOTE_DIR)

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            super().end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = index_html().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt, *args):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log("serving %s on port %d" % (NOTE_DIR, PORT))
    for ip in local_ips():
        log("  http://%s:%d/   (docs: /master.html, /notebook.html)" % (ip, PORT))


def local_ips():
    ips = set()
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.split():
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


def port_in_use():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()


def watch():
    log("watching %d documents (pid %d)" % (len(DOCS), os.getpid()))
    last = {}
    while True:
        try:
            for name, _, _ in DOCS:
                if is_stale(name) and time.time() - last.get(name, 0) > RETRY_SECONDS:
                    last[name] = time.time()
                    if build(name) and is_stale(name):
                        log("%s changed during build; rebuilding" % name)
                        last[name] = 0.0
        except Exception:
            log("! loop error:\n" + traceback.format_exc())
        time.sleep(POLL_SECONDS)


def main():
    args = sys.argv[1:]
    if "--serve" in args:
        if port_in_use():
            log("port %d already in use - another instance is running, exiting" % PORT)
            return 0
        serve()
        try:
            build_all()
        except Exception:
            log("! initial build error:\n" + traceback.format_exc())
        watch()
    elif "--watch" in args:
        try:
            build_all()
        except Exception:
            log("! initial build error:\n" + traceback.format_exc())
        watch()
    else:
        return 0 if build_all() else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
