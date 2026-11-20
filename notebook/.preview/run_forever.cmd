@echo off
rem Keep the notebook preview alive: build + serve + auto-rebuild, restart if it dies.
cd /d "%~dp0.."

:loop
netstat -ano | findstr /c:"LISTENING" | findstr /c:":8899 " >nul 2>&1
if not errorlevel 1 (
  rem someone is already serving on 8899 - wait and re-check
  timeout /t 30 /nobreak >nul
  goto loop
)
python ".preview\build.py" --serve
timeout /t 5 /nobreak >nul
goto loop
