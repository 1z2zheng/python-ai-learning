@echo off
for /d %%D in ("%~dp0C260928003*") do if exist "%%~fD\app.py" cd /d "%%~fD"
if exist "..\venv-unet\Scripts\python.exe" (
  "..\venv-unet\Scripts\python.exe" -B app.py
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -B app.py
) else (
  echo Please run install_dependencies.cmd first.
)
pause
