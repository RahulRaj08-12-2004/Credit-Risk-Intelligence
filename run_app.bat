@echo off
title Credit Risk Intelligence - Streamlit Dashboard
echo ========================================================
echo    Credit Risk Intelligence Platform (Streamlit)
echo ========================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo Starting dashboard on http://localhost:8501 ...
"C:\Users\sanka\NEW ANACONDA\python.exe" -m streamlit run app.py
pause
