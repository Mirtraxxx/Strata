@echo off
rem Updates this Strata folder from the official project (Niko1221/Strata) while keeping our own changes (branch
rem "mine", also on github.com/Mirtraxxx/Strata). Use this instead of UPDATE.bat: UPDATE.bat only does a plain
rem fast-forward pull, which stops as soon as the folder has changes of its own.
rem Steps: get the official code, merge it into ours, compile the engine, install it (Strata must be closed for
rem that), then Strata's own package/settings update without touching the engine. If the official update clashes
rem with our changes, nothing is changed and Claude does the merge by hand.
setlocal
title Strata - update (keeping our changes)
cd /d "%~dp0"
rem All of it in one block: cmd reads a .bat file while it runs it, and the merge can change files in this folder.
(
  for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH=%%b"
  call :check_branch || exit /b 1
  git diff --quiet HEAD || (
    echo.
    echo  This folder has changes that are not saved in git yet, so nothing was updated.
    echo  Ask Claude to look at them first.
    pause
    exit /b 1
  )
  echo  Getting the newest official Strata ...
  git fetch origin || (
    echo.
    echo  Could not reach GitHub ^(the reason is above^). Nothing was updated.
    pause
    exit /b 1
  )
  git merge --no-edit origin/main || (
    git merge --abort
    echo.
    echo  The official update clashes with our own changes, so nothing was changed.
    echo  Ask Claude to merge the update by hand.
    pause
    exit /b 1
  )
  echo  Saving our updated version to github.com/Mirtraxxx/Strata ...
  git push fork mine || echo  ^(could not upload it; the update itself is fine - Claude can push it later^)
  echo  Compiling the engine ^(a few minutes when it changed, seconds when not^) ...
  call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
  "%~dp0.venv\Scripts\cmake.exe" --build build-release --target strata -j 8 || (
    echo.
    echo  The engine did not compile ^(the reason is above^). The code is updated, the installed engine is unchanged.
    echo  Ask Claude to fix the build.
    pause
    exit /b 1
  )
  fc /b build-release\strata.exe engine\strata.exe >nul 2>nul && (
    echo  The engine did not change.
  ) || (
    copy /y engine\strata.exe engine\strata.exe.previous >nul 2>nul
    copy /y build-release\strata.exe engine\strata.exe >nul 2>nul && (
      echo  New engine installed ^(the old one is engine\strata.exe.previous^).
    ) || (
      echo.
      echo  The new engine could not be installed because Strata is running.
      echo  Close the Strata window and run this again: the rest is already done.
      pause
      exit /b 1
    )
  )
  echo  Updating Strata's Python packages and model settings ...
  "%~dp0.venv\Scripts\python.exe" setup.py --update --build
  echo.
  echo  Done. Start Strata the usual way.
  pause
  exit /b 0
)

:check_branch
if /i "%BRANCH%"=="mine" exit /b 0
echo.
echo  This folder is on the git branch "%BRANCH%", not "mine", so nothing was updated.
echo  Ask Claude to switch it back.
pause
exit /b 1
