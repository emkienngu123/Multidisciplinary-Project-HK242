@echo off
mkdir mp3_output
for %%I in (*.m4a) DO ffmpeg -i "%%I" "mp3_output\%%~nI.mp3"
pause