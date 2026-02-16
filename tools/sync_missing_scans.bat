@echo off
setlocal enabledelayedexpansion

REM Requires env var MRMVS_SSH_PASS to be set by caller.
set "SSH_ASKPASS=%~dp0askpass.bat"
set "SSH_ASKPASS_REQUIRE=force"
set "DISPLAY=1"
set "HOST=121.36.253.161"
set "PORT=25213"
set "USER=zan"
set "REMOTE_BASE=/mnt/remote_data/DTU/render"
set "LOCAL_BASE=static\render_compare"

if not exist "%LOCAL_BASE%" mkdir "%LOCAL_BASE%"

REM List of scans to ensure.
for %%S in (scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan9) do (
  call :sync_scan %%S
)

echo Done.
exit /b 0

:sync_scan
set "SCAN=%~1"
echo === %SCAN% ===
if not exist "%LOCAL_BASE%\%SCAN%" mkdir "%LOCAL_BASE%\%SCAN%"

call :sync_method %SCAN% transmvsnet transmvsnet
call :sync_method %SCAN% geomvsnet geomvsnet
call :sync_method %SCAN% mvsformer mvsformerplusplus
call :sync_method %SCAN% mrmvs_1 mrmvs

exit /b 0

:sync_method
set "SCAN=%~1"
set "REMOTE_METHOD=%~2"
set "LOCAL_METHOD=%~3"

if exist "%LOCAL_BASE%\%SCAN%\%LOCAL_METHOD%\*" (
  echo [skip] %SCAN%/%LOCAL_METHOD%
  exit /b 0
)

echo [get ] %SCAN%/%LOCAL_METHOD%
REM Note: avoid quoting a path that ends with '\' (scp treats it badly and may include a stray quote).
C:\Windows\System32\OpenSSH\scp.exe -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -P %PORT% -r %USER%@%HOST%:%REMOTE_BASE%/%SCAN%/%REMOTE_METHOD% "%LOCAL_BASE%\%SCAN%" < NUL
if errorlevel 1 (
  echo [miss] %SCAN%/%REMOTE_METHOD%
  exit /b 0
)

if /I not "%REMOTE_METHOD%"=="%LOCAL_METHOD%" (
  if exist "%LOCAL_BASE%\%SCAN%\%REMOTE_METHOD%" (
    if exist "%LOCAL_BASE%\%SCAN%\%LOCAL_METHOD%" rmdir /s /q "%LOCAL_BASE%\%SCAN%\%LOCAL_METHOD%"
    ren "%LOCAL_BASE%\%SCAN%\%REMOTE_METHOD%" "%LOCAL_METHOD%"
  )
)

echo [ok  ] %SCAN%/%LOCAL_METHOD%
exit /b 0
