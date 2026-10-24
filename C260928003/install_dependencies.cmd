@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo Python 3.12 is required.
  pause
  exit /b 1
)
for /d %%D in ("%~dp0C260928003*") do if exist "%%~fD\requirements.txt" ".venv\Scripts\python.exe" -m pip install -r "%%~fD\requirements.txt"
pause
