@echo off

REM Change your VsDevCmd.bat path here
set _VS_CMD="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat"

set _CD_PATH=%CD%
set _CD_DRIVE=%CD:~0,2%

%~d0
cd %~dp0
cd ..

if not defined ROS_INIT (
    call %_VS_CMD% -arch=amd64 -host_arch=amd64
    set ChocolateyInstall=c:\opt\chocolatey
    call c:\opt\ros\noetic\x64\setup.bat
    set ROS_INIT=""
)

set _SETUP_FILE="devel\setup.bat"

if not exist %_SETUP_FILE% (
    catkin_make
    set ROS_SETUP=
)

if not defined ROS_SETUP (
    call %_SETUP_FILE%
    set ROS_SETUP=""
)

%_CD_DRIVE%
cd %_CD_PATH%

set _VS_CMD=
set _CD_DRIVE=
set _CD_PATH=
set _SETUP_FILE=