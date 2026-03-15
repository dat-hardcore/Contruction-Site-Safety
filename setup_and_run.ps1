param(
    [string]$ModelPath,
    [int]$CameraIndex,
    [string]$AreaName,
    [switch]$DemoMode,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "Python was not found. Install Python 3.10+ and try again."
}

function Ensure-Venv {
    param([string]$PythonCommand, [string]$ProjectRoot)

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating virtual environment (.venv)..."
        & $PythonCommand -m venv (Join-Path $ProjectRoot ".venv")
    } else {
        Write-Host "Virtual environment already exists."
    }
    return $venvPython
}

function Ensure-Dependencies {
    param([string]$VenvPython, [string]$ProjectRoot)

    Write-Host "Installing dependencies..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

function Load-Or-CreateSettings {
    param([string]$SettingsPath)

    if (Test-Path $SettingsPath) {
        $raw = Get-Content -Raw $SettingsPath
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return [ordered]@{
                camera_index = 0
                camera_area_name = "Khu vuc A"
                confidence_threshold = 0.25
                violation_throttle_seconds = 0.8
                model_path = $null
            }
        }
        return ($raw | ConvertFrom-Json -AsHashtable)
    }

    return [ordered]@{
        camera_index = 0
        camera_area_name = "Khu vuc A"
        confidence_threshold = 0.25
        violation_throttle_seconds = 0.8
        model_path = $null
    }
}

function Normalize-ModelPath {
    param([string]$InputPath, [string]$ProjectRoot)

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        $fullPath = [System.IO.Path]::GetFullPath($InputPath)
    } else {
        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $InputPath))
    }

    if (-not (Test-Path $fullPath)) {
        throw "Model file not found: $fullPath"
    }

    return $fullPath
}

$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

$pythonCommand = Get-PythonCommand
$venvPython = Ensure-Venv -PythonCommand $pythonCommand -ProjectRoot $projectRoot

if (-not $SkipInstall) {
    Ensure-Dependencies -VenvPython $venvPython -ProjectRoot $projectRoot
} else {
    Write-Host "Skipping dependency installation."
}

$dataDir = Join-Path $projectRoot "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}
$settingsPath = Join-Path $dataDir "settings.json"
$settings = Load-Or-CreateSettings -SettingsPath $settingsPath

if ($PSBoundParameters.ContainsKey("CameraIndex")) {
    $settings["camera_index"] = $CameraIndex
}

if ($PSBoundParameters.ContainsKey("AreaName")) {
    $settings["camera_area_name"] = $AreaName
}

if ($DemoMode) {
    $settings["model_path"] = $null
    Write-Host "Demo mode enabled: model_path = null"
} elseif ($PSBoundParameters.ContainsKey("ModelPath")) {
    $settings["model_path"] = Normalize-ModelPath -InputPath $ModelPath -ProjectRoot $projectRoot
}

$settingsJson = $settings | ConvertTo-Json -Depth 5
Set-Content -Path $settingsPath -Value $settingsJson -Encoding UTF8
Write-Host "Updated settings: $settingsPath"

Write-Host "Starting app..."
& $venvPython (Join-Path $projectRoot "run.py")
