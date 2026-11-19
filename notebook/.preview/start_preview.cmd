@echo off
rem Start notebook preview: build HTML+PDF, watch notebook.md, serve over Tailscale.
cd /d "%~dp0.."
python ".preview\build.py" --serve
