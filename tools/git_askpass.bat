@echo off
set PROMPT_TEXT=%1
echo %PROMPT_TEXT% | C:\Windows\System32\findstr.exe /I "Username" >NUL
if %ERRORLEVEL%==0 (
  echo %GH_USERNAME%
  exit /b 0
)
echo %GH_TOKEN%
