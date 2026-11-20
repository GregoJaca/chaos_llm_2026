"""Build notebook_big.md -> notebook_big.html (offline KaTeX) -> notebook_big.pdf
(headless Chrome), optionally serve over HTTP and rebuild automatically whenever
notebook_big.md changes.

Usage:
  python .preview/build.py            # build once
  python .preview/build.py --watch    # rebuild on change
  python .preview/build.py --serve    # build, serve over HTTP (Tailscale), keep watching

Self-healing: the loop rebuilds whenever notebook_big.md is newer than notebook_big.pdf,
so a missed event, a crashed browser, a killed process or a laptop restart all recover by
themselves. Every run is logged to .preview/build.log.
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
SRC = os.path.join(NOTE_DIR, "notebook_big.md")
HTML_OUT = os.path.join(NOTE_DIR, "notebook_big.html")
PDF_OUT = os.path.join(NOTE_DIR, "notebook_big.pdf")
LOG_PATH = os.path.join(HERE, "build.log")
KATEX = os.path.join(HERE, "katex")
PORT = int(os.environ.get("NOTEBOOK_PORT", "8898"))

POLL_SECONDS = 2.0          # how often to check notebook_big.md
RETRY_SECONDS = 20.0        # back-off after a failed build

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


# --------------------------------------------------------------------------- utils

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


def is_stale():
    """True when notebook_big.md is newer than the built PDF (or the PDF is missing)."""
    return mtime(SRC) > mtime(PDF_OUT) or mtime(SRC) > mtime(HTML_OUT)


# --------------------------------------------------------------------------- build

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
    """Render markdown -> html -> pdf. Returns True on success, never raises."""
    import markdown

    t0 = time.time()
    try:
        with open(SRC, "r", encoding="utf-8") as f:
            md = f.read()
    except Exception as e:
        log("! cannot read %s: %s" % (SRC, e))
        return False

    try:
        body_md, math = protect_math(md)
        body = markdown.markdown(
            body_md,
            extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
        )
        body = restore_math(body, math)
    except Exception:
        log("! markdown error:\n" + traceback.format_exc())
        return False

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>notebook_big.md</title>
<link rel="stylesheet" href=".preview/katex/katex.min.css">
<style>{CSS}</style>
</head><body>
<header class="banner">
  built {stamp} &nbsp;&middot;&nbsp; source: notebook_big.md &nbsp;&middot;&nbsp;
  <a href="notebook_big.pdf">PDF</a> &nbsp;&middot;&nbsp; <a href="notebook_big.md">Markdown</a>
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
        tmp_html = HTML_OUT + ".tmp"
        with open(tmp_html, "w", encoding="utf-8") as f:
            f.write(page)
        os.replace(tmp_html, HTML_OUT)
    except Exception:
        log("! html write error:\n" + traceback.format_exc())
        return False

    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        log("! no Edge/Chrome found; wrote HTML only")
        return False

    tmp_pdf = PDF_OUT + ".tmp.pdf"
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
        "--virtual-time-budget=20000",
        f"--print-to-pdf={tmp_pdf}",
        "file:///" + HTML_OUT.replace("\\", "/"),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log("! browser timed out after 300s")
        return False
    except Exception:
        log("! browser launch failed:\n" + traceback.format_exc())
        return False

    if os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 10_000:
        # sanity: a healthy PDF is much bigger than a blank page
        try:
            os.replace(tmp_pdf, PDF_OUT)
        except Exception:
            log("! cannot replace pdf: %s" % traceback.format_exc())
            return False
    else:
        tail = (r.stderr or r.stdout or "")[-600:]
        log("! pdf failed (exit %s): %s" % (r.returncode, tail.replace("\n", " ")))
        try:
            os.remove(tmp_pdf)
        except OSError:
            pass
        return False

    log("built %s  html %.0f kB  pdf %.1f MB  in %.1fs"
        % (stamp, os.path.getsize(HTML_OUT) / 1e3,
           os.path.getsize(PDF_OUT) / 1e6, time.time() - t0))
    return True


# --------------------------------------------------------------------------- server

INDEX = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>notebook (large models)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;margin:0;padding:28px 20px;
max-width:640px;margin:0 auto;color:#111;background:#fafafa}
h1{font-size:1.35em;margin:0 0 4px}
p.meta{color:#666;font-size:.85em;margin:0 0 26px}
a.card{display:block;text-decoration:none;color:inherit;background:#fff;border:1px solid #ddd;
border-radius:10px;padding:16px 18px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
a.card b{display:block;font-size:1.05em;margin-bottom:3px}
a.card span{color:#666;font-size:.85em}
</style></head><body>
<h1>notebook (large models)</h1>
<p class="meta">%s</p>
<a class="card" href="notebook_big.pdf"><b>PDF (newest)</b><span>Download / open &mdash; equations and figures included</span></a>
<a class="card" href="notebook_big.html"><b>HTML</b><span>Scroll online</span></a>
<a class="card" href="notebook_big.md"><b>Markdown</b><span>Raw source</span></a>
</body></html>
"""


def serve():
    os.chdir(NOTE_DIR)

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            super().end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (INDEX % (
                    "notebook_big.md last edited %s"
                    % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime(SRC)))
                )).encode()
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
        log("  http://%s:%d/   (pdf: /notebook_big.pdf)" % (ip, PORT))


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


# --------------------------------------------------------------------------- loop

def watch(serve_mode=False):
    log("watching %s (pid %d)" % (SRC, os.getpid()))
    last_attempt = 0.0
    while True:
        try:
            if is_stale() and time.time() - last_attempt > RETRY_SECONDS:
                last_attempt = time.time()
                if build():
                    # guard against a file edited *during* the build
                    if is_stale():
                        log("source changed during build; rebuilding")
                        last_attempt = 0.0
        except Exception:
            log("! loop error:\n" + traceback.format_exc())
        time.sleep(POLL_SECONDS)


def main():
    args = sys.argv[1:]
    if "--serve" in args:
        if port_in_use():
            log("port %d already in use - another instance is running, exiting" % PORT)
            return 0
        serve()                      # serve the previous PDF immediately
        try:
            build()                  # then refresh it
        except Exception:
            log("! initial build error:\n" + traceback.format_exc())
        watch(serve_mode=True)
    elif "--watch" in args:
        try:
            build()
        except Exception:
            log("! initial build error:\n" + traceback.format_exc())
        watch()
    else:
        return 0 if build() else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
