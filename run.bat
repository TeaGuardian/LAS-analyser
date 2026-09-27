@echo off
REM Активируем виртуальное окружение
call "%~dp0venv\Scripts\activate.bat"

REM Запускаем background.py с помощью python из venv
python "%~dp0app.py"

REM По желанию, деактивируем окружение (не обязательно)
call deactivate

pause
