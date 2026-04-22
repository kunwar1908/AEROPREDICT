@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo ==================================
echo RUL Prediction Pipeline
echo ==================================
echo Using Python: %PYTHON_BIN%

echo.
echo 1. Downloading official NASA C-MAPSS data...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\download_data.py"
if errorlevel 1 goto :error

echo.
echo 2. Training model on FD001...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\train.py" --dataset FD001
if errorlevel 1 goto :error

echo.
echo 3. Evaluating saved checkpoint...
"%PYTHON_BIN%" "%SCRIPT_DIR%src\evaluate.py" --dataset FD001
if errorlevel 1 goto :error

echo.
echo ==================================
echo Pipeline completed successfully!
echo ==================================
goto :eof

:error
echo.
echo Pipeline failed.
exit /b 1
