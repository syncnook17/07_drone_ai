@echo off
cd /d %~dp0\..
call venv\Scripts\activate
pyinstaller build\DroneCVEdu.spec --noconfirm --clean
echo.
echo เสร็จ - ผลอยู่ที่ dist\DroneCVEdu.exe  (ไฟล์เดียว ดับเบิลคลิกเปิดได้เลย)
pause
