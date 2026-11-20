@echo off
chcp 65001 >nul
title Prism-v3-exp - inference webui
setlocal
cd /d "%~dp0.."

rem 只依赖 onnxruntime + numpy，不需要 torch：
rem   pip install onnxruntime numpy
rem 想用完整 .pt 权重（带续写）就改用：
rem   python src\webui.py --checkpoint <prism-v3-exp.pt> --open

set "MODEL=models\domain_int8.onnx"
set "META=models\model_meta.json"
set "PORT=8902"

echo.
echo   =========================================================
echo    Prism-v3-exp  /  inference webui
echo   ---------------------------------------------------------
echo    model : %MODEL%
echo    url   : http://127.0.0.1:%PORT%
echo   ---------------------------------------------------------
echo    Keep this window open. Close it (or press Ctrl+C)
echo    to stop the service.
echo   =========================================================
echo.

if not exist "%MODEL%" goto noModel

netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul
if not errorlevel 1 goto alreadyRunning

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python src\serve_onnx.py --model "%MODEL%" --meta "%META%" --port %PORT% --host 127.0.0.1 %*
echo.
echo service stopped.
pause
exit /b 0

:alreadyRunning
echo port %PORT% is already in use - opening browser instead.
start "" "http://127.0.0.1:%PORT%"
timeout /t 2 >nul
exit /b 0

:noModel
echo [ERROR] model not found:
echo         %MODEL%
echo clone the repo fully, or export it yourself:
echo         python tools\export_onnx.py --out data\onnx
pause
exit /b 1
