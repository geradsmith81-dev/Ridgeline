@echo off
rem Puts an "Ridgeline" icon on the Desktop and in the Start menu that opens the app.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcut.ps1"
echo.
echo  The Ridgeline icon is now on your Desktop and in the Start menu.
echo.
pause
