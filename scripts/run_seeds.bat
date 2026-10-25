@echo off
REM Same as run_seeds.ps1, callable from cmd / Anaconda Prompt:
REM   scripts\run_seeds.bat -Runs 3 -Steps 1000000 -EvalInterval 50000
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_seeds.ps1" %*
