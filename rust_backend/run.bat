@echo off
chcp 65001 >nul
cd /d "%~dp0"
title PhonePicker

where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust не найден.
  echo Установите его командой:  winget install Rustlang.Rustup
  echo или скачайте установщик с https://rustup.rs
  echo После установки закройте это окно и запустите run.bat ещё раз.
  pause
  exit /b 1
)

echo Сборка (первый раз 1-3 минуты, потом пара секунд)...
cargo build --release
if errorlevel 1 (
  echo Сборка не удалась, текст ошибки выше.
  pause
  exit /b 1
)

start "PhonePicker server" target\release\phonepicker-server.exe
timeout /t 2 /nobreak >nul
start "" http://localhost:8080
echo Сервер запущен в отдельном окне: http://localhost:8080
echo Чтобы остановить, закройте окно "PhonePicker server".
timeout /t 5 >nul
