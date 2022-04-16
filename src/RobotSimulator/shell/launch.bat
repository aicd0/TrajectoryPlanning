@echo off

call "%~dp0/setup.bat"

roslaunch robot_sim main.launch gui:=true