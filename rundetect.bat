@echo off
IF NOT EXIST %~dp0\footage (
	mkdir %~dp0\footage
)
cd /d %~dp0
python motion_detector.py


