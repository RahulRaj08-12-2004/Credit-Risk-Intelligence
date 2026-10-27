@echo off
title Credit Risk Intelligence - Test Suite
echo ========================================================
echo          Running Automated Pytest Test Suite
echo ========================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
"C:\Users\sanka\NEW ANACONDA\python.exe" -m pytest tests -v
pause
