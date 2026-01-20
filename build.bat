@ECHO OFF

:: 1. Sprawdzenie uprawnień i wymuszenie admina
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :admin
) else (
    echo Prosba o uprawnienia administratora...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:admin
:: 2. Ustawienie folderu roboczego (naprawia problem uruchamiania jako admin)
cd /d "%~dp0"

echo [1/4] Usuwanie starych plikow...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [2/4] Kompilacja aplikacji...
pyinstaller --onefile --noconsole --icon=assets/icon.png --collect-all mediapipe main.py

echo [3/4] Kopiowanie folderow do folderu dist...
:: Usunieto 'rem', aby foldery faktycznie sie kopiowaly
xcopy "assets" "dist\assets" /E /I /Y
xcopy "dataset" "dist\dataset" /E /I /Y
xcopy "save" "dist\save" /E /I /Y
xcopy "trained_models" "dist\trained_models" /E /I /Y

echo [4/4] Sprzatanie...
if exist build rmdir /s /q build
if exist main.spec del /q main.spec

echo.s
echo ZAKONCZONO POMYSLNIE!
pause