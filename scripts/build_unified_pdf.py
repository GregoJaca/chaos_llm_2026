"""Build unified.md -> unified.html (KaTeX) -> unified.pdf (headless Edge/Chrome).
Also copies unified.pdf and unified.html to notebook_big/ for immediate Tailscale serving.
"""

import os
import re
import shutil
import subprocess
import time
import traceback
import markdown

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "unified.md")
HTML_OUT = os.path.join(REPO_ROOT, "unified.html")
PDF_OUT = os.path.join(REPO_ROOT, "unified.pdf")
KATEX_DIR = os.path.join(REPO_ROOT, "notebook_big", ".preview", "katex")
NOTEBOOK_BIG_DIR = os.path.join(REPO_ROOT, "notebook_big")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
:root { --fg:#1a1a1a; --muted:#555; --line:#d8d8d8; --accent:#0b5cad; }
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Cambria", serif;
  font-size: 14.5px; line-height: 1.55; color: var(--fg);
  margin: 0 auto; max-width: 216mm; padding: 18mm 16mm 24mm;
  background: #fff;
}
h1, h2, h3, h4 { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; line-height:1.25; }
h1 { font-size: 1.65em; margin: 1.2em 0 .4em; border-bottom: 2px solid var(--line); padding-bottom:.25em; color:#0b3c6d; }
h2 { font-size: 1.28em; margin: 1.2em 0 .35em; color:#111; border-bottom: 1px solid #e5e5e5; padding-bottom: 3px; }
h3 { font-size: 1.05em; margin: 1.0em 0 .25em; color:#222; }
p, li { margin: .4em 0; }
ul, ol { margin: .4em 0 .4em 1.3em; padding: 0; }
code, pre { font-family: "Cascadia Mono", Consolas, monospace; font-size: .84em; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }
pre { background:#f7f7f7; border:1px solid #e5e5e5; border-radius:4px; padding:8px 10px; overflow-x:auto; }
img { max-width: 100%; height: auto; display:block; margin: 14px auto; border: 1px solid #e2e8f0; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: .90em; }
th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f5f7f9; font-weight: 600; }
hr { border: 0; border-top: 1px solid var(--line); margin: 1.6em 0; }
blockquote { margin: .6em 0; padding: .4em 1em; border-left: 3px solid #0b5cad; background: #f8fafc; color:#333; }
a { color: var(--accent); }
.katex-display { margin: .65em 0; overflow-x: auto; overflow-y: hidden; }
.katex { font-size: 1.02em; }
header.banner {
  font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; color: var(--muted);
  border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 18px;
}
header.banner a { text-decoration: none; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
img, table, pre, .katex-display { break-inside: avoid; page-break-inside: avoid; }
@page { size: A4; margin: 14mm 14mm 15mm; }
"""

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
    return re.sub(r"\u2063MATH(\d+)\u2063", repl, html)

def main():
    print("Reading unified.md...")
    with open(SRC, "r", encoding="utf-8") as f:
        md = f.read()

    body_md, math = protect_math(md)
    body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
    )
    body = restore_math(body, math)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    katex_rel = os.path.relpath(KATEX_DIR, REPO_ROOT).replace("\\", "/")

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unified Cross-Scale Dynamics & Stability Deliverable</title>
<link rel="stylesheet" href="{katex_rel}/katex.min.css">
<style>{CSS}</style>
</head><body>
<header class="banner">
  Built {stamp} &nbsp;&middot;&nbsp; Source: unified.md &nbsp;&middot;&nbsp;
  <a href="unified.pdf">Download PDF</a> &nbsp;&middot;&nbsp; <a href="unified.md">Markdown Source</a>
</header>
{body}
<script src="{katex_rel}/katex.min.js"></script>
<script src="{katex_rel}/contrib/auto-render.min.js"></script>
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
    print(f"Wrote {HTML_OUT} ({os.path.getsize(HTML_OUT):,} bytes)")

    # Find browser
    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        print("ERROR: No Edge/Chrome executable found.")
        return

    print(f"Compiling PDF with {exe}...")
    tmp_pdf = PDF_OUT + ".tmp.pdf"
    if os.path.exists(tmp_pdf):
        os.remove(tmp_pdf)

    file_url = "file:///" + HTML_OUT.replace("\\", "/")
    cmd = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={tmp_pdf}",
        file_url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 20000:
        if os.path.exists(PDF_OUT):
            os.remove(PDF_OUT)
        os.replace(tmp_pdf, PDF_OUT)
        print(f"Generated {PDF_OUT} ({os.path.getsize(PDF_OUT):,} bytes)")
    else:
        print("PDF compilation failed:", r.stderr or r.stdout)
        return

    # Copy PDF and HTML to notebook_big/ for serving
    dest_pdf = os.path.join(NOTEBOOK_BIG_DIR, "unified.pdf")
    shutil.copy2(PDF_OUT, dest_pdf)
    dest_md = os.path.join(NOTEBOOK_BIG_DIR, "unified.md")
    shutil.copy2(SRC, dest_md)
    print(f"Copied to {dest_md} for Tailscale HTTP serving.")

    dest_html = os.path.join(NOTEBOOK_BIG_DIR, "unified.html")
    # For HTML in notebook_big, katex is at .preview/katex
    page_nb = page.replace(f"{katex_rel}/", ".preview/katex/").replace('src="unified_figures/', 'src="../unified_figures/')
    with open(dest_html, "w", encoding="utf-8") as f:
        f.write(page_nb)
    print(f"Copied to {dest_html}.")

    # Also ensure unified_figures are accessible from notebook_big
    nb_figdir = os.path.join(NOTEBOOK_BIG_DIR, "unified_figures")
    if not os.path.exists(nb_figdir):
        try:
            # Create directory and copy pngs
            os.makedirs(nb_figdir, exist_ok=True)
            src_figdir = os.path.join(REPO_ROOT, "unified_figures")
            for f in os.listdir(src_figdir):
                if f.endswith(".png") or f.endswith(".json"):
                    shutil.copy2(os.path.join(src_figdir, f), os.path.join(nb_figdir, f))
            print(f"Mirrored unified_figures into {nb_figdir}")
        except Exception as e:
            print("Note: could not mirror unified_figures:", e)
    else:
        src_figdir = os.path.join(REPO_ROOT, "unified_figures")
        for f in os.listdir(src_figdir):
            if f.endswith(".png") or f.endswith(".json"):
                shutil.copy2(os.path.join(src_figdir, f), os.path.join(nb_figdir, f))

    print("\nSUCCESS: unified.pdf compiled and served!")

if __name__ == "__main__":
    main()
