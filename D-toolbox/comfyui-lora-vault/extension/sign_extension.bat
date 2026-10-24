@echo off
rem Signs the extension with Mozilla (free, "unlisted" = private, not published) so it installs
rem PERMANENTLY in normal Firefox. Needs Node.js and AMO API keys:
rem   https://addons.mozilla.org/developers/addon/api/key/
rem Usage:  sign_extension.bat  YOUR_JWT_ISSUER  YOUR_JWT_SECRET
if "%~2"=="" (
  echo Usage: sign_extension.bat JWT_ISSUER JWT_SECRET
  exit /b 1
)
cd /d "%~dp0"
call npx --yes web-ext@8 sign --channel=unlisted --source-dir=. --artifacts-dir=signed ^
  --ignore-files "sign_extension.bat" "signed/**" --api-key=%1 --api-secret=%2
echo.
echo Done. The signed file is in the "signed" folder (lora_vault_for_civitai-*.xpi).
echo In Firefox: open about:addons, click the gear, "Install Add-on From File...", pick that .xpi.
pause
