@echo off
REM Download a small test dataset for development.
REM
REM Usage:
REM   scripts\download_dataset.bat [dataset] [output_dir]
REM
REM Datasets:
REM   truck   — Tanks & Temples Truck scene (~250 images, ~500 MB)
REM   train   — Tanks & Temples Train scene (~300 images, ~600 MB)
REM
REM Requires: curl (ships with Windows 10+), tar (ships with Windows 10+)
REM
REM The Tanks & Temples scenes are hosted by the 3DGS authors at:
REM   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
REM
REM This downloads the full Tanks & Temples + Deep Blending bundle (~1.4 GB).
REM Individual scene downloads are not available separately.

setlocal

set DATASET=%~1
if "%DATASET%"=="" set DATASET=truck

set OUTPUT_DIR=%~2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=data

set URL=https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
set ZIP_FILE=%OUTPUT_DIR%\tandt_db.zip

echo ============================================================
echo  3D Gaussian Splatting — Test Dataset Downloader
echo ============================================================
echo.
echo Dataset URL: %URL%
echo Output dir:  %OUTPUT_DIR%
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Download if not already present
if exist "%ZIP_FILE%" (
    echo ZIP already downloaded: %ZIP_FILE%
) else (
    echo Downloading Tanks ^& Temples + Deep Blending dataset...
    echo This is ~1.4 GB — it may take a few minutes.
    echo.
    curl -L -o "%ZIP_FILE%" "%URL%"
    if errorlevel 1 (
        echo ERROR: Download failed.
        exit /b 1
    )
)

REM Extract
echo.
echo Extracting...
tar -xf "%ZIP_FILE%" -C "%OUTPUT_DIR%"
if errorlevel 1 (
    echo ERROR: Extraction failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Available scenes in %OUTPUT_DIR%:
echo ============================================================
dir /b /ad "%OUTPUT_DIR%\tandt" 2>nul
dir /b /ad "%OUTPUT_DIR%\db" 2>nul
echo.
echo Example usage:
echo   build\dump_points %OUTPUT_DIR%\tandt\truck
echo   build\dump_points %OUTPUT_DIR%\tandt\train
echo.

endlocal
