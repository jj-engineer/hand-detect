# =============================================================================
#  TITAN-9  -  Windows one-click setup + launcher
#
#  Started by START_WINDOWS.bat (double-click it).  What it does, in order:
#    1. finds a 64-bit Python 3.9-3.12, or downloads and installs Python 3.11 (no admin needed)
#    2. makes sure the Microsoft Visual C++ runtime is present (needed by OpenCV / MediaPipe)
#    3. builds its own virtual environment (.venv-win) and installs requirements-windows.txt
#    4. verifies every package really imports (rebuilds the environment once if not)
#    5. checks the laptop camera, camera privacy switch, MediaPipe model and OpenGL 4.1
#    6. launches the game
#
#  Options (give them to START_WINDOWS.bat):   --reinstall   rebuild the environment from scratch
#                                              --setup-only  install everything but do not start the game
#                                              --check       run the camera / graphics check only
#  Any other option (--demo, --no-camera, --quality medium, --camera 1, ...) is passed on to the game.
#
#  NOTE: keep this file ASCII-only - Windows PowerShell 5.1 reads BOM-less files as ANSI.
# =============================================================================

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'     # progress bars make Invoke-WebRequest extremely slow in PowerShell 5.1
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$LogFile   = Join-Path $Root 'setup_windows.log'
$ReqFile   = Join-Path $Root 'requirements-windows.txt'
$SelfCheck = Join-Path $Root 'tools\windows_selfcheck.py'
$PythonToInstall = '3.11.9'
$script:AppExit  = 0

# ---- options -----------------------------------------------------------------
$Reinstall = $false; $SetupOnly = $false; $CheckOnly = $false; $AppArgs = @()
if ($env:TITAN_ARGS) {
    foreach ($a in @($env:TITAN_ARGS -split '\s+' | Where-Object { $_ })) {
        switch ($a.ToLower()) {
            '--reinstall'  { $Reinstall = $true }
            '--setup-only' { $SetupOnly = $true }
            '--check'      { $CheckOnly = $true }
            default        { $AppArgs += $a }
        }
    }
}

# ---- output helpers ----------------------------------------------------------
function Say([string]$msg, [string]$color = 'Gray') { Write-Host $msg -ForegroundColor $color }
function Step([int]$n, [int]$total, [string]$msg) { Write-Host ''; Write-Host ("[{0}/{1}] {2}" -f $n, $total, $msg) -ForegroundColor Cyan }
function Ok([string]$msg)   { Write-Host ("      OK  " + $msg) -ForegroundColor Green }
function Warn([string]$msg) { Write-Host ("      !!  " + $msg) -ForegroundColor Yellow }
function Die([string]$msg)  { throw $msg }

# ---- download helpers --------------------------------------------------------
function Get-Download([string]$url, [string]$dest) {
    for ($try = 1; $try -le 3; $try++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $dest -ErrorAction Stop
            if ((Test-Path -LiteralPath $dest) -and ((Get-Item -LiteralPath $dest).Length -gt 1MB)) { return $true }
        } catch {
            Warn ("download attempt {0} of 3 failed: {1}" -f $try, $_.Exception.Message)
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

# A tampered download is rejected; an unknown-root / offline-revocation result is tolerated (the download itself was HTTPS).
function Test-DownloadTrusted([string]$file) {
    try {
        $sig = Get-AuthenticodeSignature -FilePath $file
        if ($sig.Status -eq 'HashMismatch' -or $sig.Status -eq 'NotSigned') { return $false }
    } catch { }
    return $true
}

# ---- Python discovery / install ----------------------------------------------
function Get-PythonInfo([string]$exe) {
    if (-not $exe) { return $null }
    if (-not (Test-Path -LiteralPath $exe)) { return $null }
    if ($exe -match '\\WindowsApps\\') { return $null }     # Microsoft Store stub: running it just opens the Store
    try {
        $out = & $exe -c "import sys,struct;print(sys.version_info[0], sys.version_info[1], struct.calcsize('P')*8)" 2>$null
    } catch { return $null }
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    $p = ("$out").Trim() -split '\s+'
    if ($p.Count -lt 3) { return $null }
    return [pscustomobject]@{ Path = $exe; Major = [int]$p[0]; Minor = [int]$p[1]; Bits = [int]$p[2] }
}

function Find-Python {
    $cands = New-Object System.Collections.ArrayList
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($v in '3.11', '3.12', '3.10', '3.9') {
            $p = & $launcher.Source "-$v" -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { [void]$cands.Add(("$p").Trim()) }
        }
    }
    foreach ($v in '311', '312', '310', '39') {
        [void]$cands.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"))
        [void]$cands.Add((Join-Path $env:ProgramFiles "Python$v\python.exe"))
        [void]$cands.Add("C:\Python$v\python.exe")
    }
    foreach ($n in 'python.exe', 'python3.exe') {
        Get-Command $n -All -ErrorAction SilentlyContinue | ForEach-Object { [void]$cands.Add($_.Source) }
    }
    $rankOf = @{ 11 = 0; 12 = 1; 10 = 2; 9 = 3 }           # mediapipe 0.10.14 supports 3.9-3.12; 3.11 is the best tested
    $best = $null; $bestRank = 99
    foreach ($c in ($cands | Select-Object -Unique)) {
        $info = Get-PythonInfo $c
        if (-not $info) { continue }
        if ($info.Major -ne 3 -or $info.Bits -ne 64 -or -not $rankOf.ContainsKey($info.Minor)) { continue }
        if ($rankOf[$info.Minor] -lt $bestRank) { $best = $info; $bestRank = $rankOf[$info.Minor] }
    }
    return $best
}

function Install-Python {
    $exe = Join-Path $env:TEMP ("python-{0}-amd64.exe" -f $PythonToInstall)
    Say ("      downloading Python {0} from python.org (about 25 MB) ..." -f $PythonToInstall)
    $done = $false
    if ((Get-Download ("https://www.python.org/ftp/python/{0}/python-{0}-amd64.exe" -f $PythonToInstall) $exe) -and (Test-DownloadTrusted $exe)) {
        Say '      installing Python for your user account (no administrator rights needed) ...'
        $p = Start-Process -FilePath $exe -Wait -PassThru -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0', 'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Shortcuts=0', 'AssociateFiles=0'
        if ($p.ExitCode -eq 0 -or $p.ExitCode -eq 3010) { $done = $true } else { Warn ("Python installer returned code {0}" -f $p.ExitCode) }
    }
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
    if (-not $done) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if ($winget) {
            Warn 'python.org route failed - trying winget instead ...'
            & $winget.Source install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Host
        }
    }
}

# ---- Visual C++ runtime ------------------------------------------------------
function Test-VcRuntime {
    $sys32 = Join-Path $env:SystemRoot 'System32'
    return ((Test-Path -LiteralPath (Join-Path $sys32 'vcruntime140_1.dll')) -and (Test-Path -LiteralPath (Join-Path $sys32 'msvcp140.dll')))
}

function Install-VcRuntime {
    $exe = Join-Path $env:TEMP 'vc_redist.x64.exe'
    Say '      downloading Microsoft Visual C++ runtime (about 25 MB) ...'
    if (-not ((Get-Download 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $exe) -and (Test-DownloadTrusted $exe))) {
        Warn 'could not download the Visual C++ runtime - continuing anyway'
        return
    }
    Say '      installing it - Windows will ask for permission, please click YES ...'
    try {
        $p = Start-Process -FilePath $exe -Wait -PassThru -Verb RunAs -ArgumentList '/install', '/quiet', '/norestart'
        if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010 -and $p.ExitCode -ne 1638) { Warn ("runtime installer returned code {0}" -f $p.ExitCode) }
    } catch {
        Warn 'the permission prompt was declined - continuing anyway (if the game fails to start, run this again and click YES)'
    }
    Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
}

# ---- camera privacy switch ---------------------------------------------------
function Get-CameraBlocks {
    $base = 'SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam'
    $found = @()
    foreach ($hive in 'HKCU', 'HKLM') {
        foreach ($sub in '', 'NonPackaged') {
            $path = "${hive}:\$base"
            if ($sub) { $path = "${hive}:\$base\$sub" }
            try {
                $val = (Get-ItemProperty -LiteralPath $path -Name Value -ErrorAction Stop).Value
                if ($val -eq 'Deny') { $found += [pscustomobject]@{ Hive = $hive; Path = $path } }
            } catch { }
        }
    }
    return $found
}

function Confirm-CameraAccess {
    $blocks = @(Get-CameraBlocks)
    if ($blocks.Count -eq 0) { Ok 'Windows camera privacy: desktop apps are allowed'; return }
    $mine = @($blocks | Where-Object { $_.Hive -eq 'HKCU' })
    $machine = @($blocks | Where-Object { $_.Hive -eq 'HKLM' })
    Warn 'Windows is blocking camera access for desktop apps (Settings > Privacy & security > Camera).'
    if ($mine.Count -gt 0) {
        $answer = Read-Host '      Turn camera access ON for this account now? [Y/n]'
        if (-not $answer -or $answer -match '^[Yy]') {
            foreach ($b in $mine) {
                try { Set-ItemProperty -LiteralPath $b.Path -Name Value -Value 'Allow' -ErrorAction Stop } catch { Warn ("could not change {0}" -f $b.Path) }
            }
            Ok 'camera access switched on'
        }
    }
    if ($machine.Count -gt 0) {
        Warn 'The camera is also blocked for the whole computer (administrator / school / work policy).'
        Warn 'Open Settings > Privacy & security > Camera and switch on "Camera access" and "Let desktop apps access your camera".'
        try { Start-Process 'ms-settings:privacy-webcam' } catch { }
    }
}

# ---- virtual environment -----------------------------------------------------
function Get-VenvDir {
    $dir = Join-Path $Root '.venv-win'
    if ($dir -match '[^\x00-\x7F]') {
        # MediaPipe cannot load its model files from paths with non-English characters
        $dir = Join-Path $env:ProgramData 'TITAN9\venv'
        Warn ("the project path has non-English characters - keeping the Python environment in {0}" -f $dir)
    }
    return $dir
}

function Test-VenvWorks([string]$venvPy) {
    if (-not (Test-Path -LiteralPath $venvPy)) { return $false }
    & $venvPy -c "import sys" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Pip([string[]]$pipArgs) {
    for ($try = 1; $try -le 3; $try++) {
        & $script:VenvPy -m pip @pipArgs | Out-Host
        if ($LASTEXITCODE -eq 0) { return $true }
        Warn ("pip did not finish (attempt {0} of 3) - trying again ..." -f $try)
        Start-Sleep -Seconds 3
    }
    return $false
}

function Install-Packages($py) {
    if (-not (Test-VenvWorks $script:VenvPy)) {
        if (Test-Path -LiteralPath $script:VenvDir) { Remove-Item -LiteralPath $script:VenvDir -Recurse -Force -ErrorAction SilentlyContinue }
        Say '      creating the private Python environment ...'
        & $py.Path -m venv $script:VenvDir | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $script:VenvPy)) { Die 'could not create the Python environment (is the disk full, or antivirus blocking the folder?)' }
    }
    Say '      updating pip ...'
    [void](Invoke-Pip @('install', '--upgrade', '--disable-pip-version-check', 'pip'))
    Say '      installing packages: MediaPipe, OpenCV, NumPy, SciPy, ModernGL, GLFW, Pillow, SoundDevice (large download - please wait) ...'
    $ok = Invoke-Pip @('install', '--disable-pip-version-check', '--no-input', '--only-binary=:all:', '--retries', '5', '--timeout', '60', '-r', $ReqFile)
    if (-not $ok) { Die 'package download failed - check the internet connection (and that antivirus / a proxy is not blocking pip), then run START_WINDOWS.bat again' }
}

# ---- main --------------------------------------------------------------------
function Invoke-Main {
    $total = 6
    Say ''
    Say '  ==============================================================' 'Cyan'
    Say '    TITAN-9  -  Windows setup and launcher' 'Cyan'
    Say '  ==============================================================' 'Cyan'
    Say '  The first run downloads and installs everything it needs'
    Say '  (about 10-20 minutes). Later runs start in a few seconds.'
    Say '  Please keep this window open.'

    if (-not [Environment]::Is64BitOperatingSystem) { Die 'TITAN-9 needs 64-bit Windows 10 or 11.' }
    if (-not (Test-Path -LiteralPath $ReqFile)) { Die ("requirements-windows.txt was not found next to this script ({0}). Extract the whole ZIP first, then run START_WINDOWS.bat from the extracted folder." -f $Root) }
    if (-not (Test-Path -LiteralPath $SelfCheck)) { Die ("tools\windows_selfcheck.py is missing - the project folder is incomplete ({0})." -f $Root) }

    # 1 --- Python
    Step 1 $total 'Python'
    $py = Find-Python
    if (-not $py) {
        Say '      no suitable Python found (64-bit Python 3.9 - 3.12 is needed) - installing it automatically'
        Install-Python
        $py = Find-Python
        if (-not $py) { Die "Python could not be installed automatically. Install 'Python 3.11 (64-bit)' from https://www.python.org/downloads/windows/ (tick nothing special), then run START_WINDOWS.bat again." }
    }
    Ok ("Python {0}.{1} (64-bit)  {2}" -f $py.Major, $py.Minor, $py.Path)

    # 2 --- Visual C++ runtime
    Step 2 $total 'Microsoft Visual C++ runtime'
    if (Test-VcRuntime) { Ok 'already installed' } else { Install-VcRuntime; if (Test-VcRuntime) { Ok 'installed' } }

    # 3 --- environment + packages
    Step 3 $total 'Python packages'
    $script:VenvDir = Get-VenvDir
    $script:VenvPy  = Join-Path $script:VenvDir 'Scripts\python.exe'
    $stampFile = Join-Path $script:VenvDir 'titan_requirements.sha256'
    $reqHash = (Get-FileHash -LiteralPath $ReqFile -Algorithm SHA256).Hash
    if ($Reinstall -and (Test-Path -LiteralPath $script:VenvDir)) {
        Say '      --reinstall: removing the old environment ...'
        Remove-Item -LiteralPath $script:VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    $upToDate = $false
    if (Test-VenvWorks $script:VenvPy) {
        $stamp = Get-Content -LiteralPath $stampFile -Raw -ErrorAction SilentlyContinue
        if ($stamp -and $stamp.Trim() -eq $reqHash) { $upToDate = $true }
    }
    if ($upToDate) {
        Ok 'packages already installed'
    } else {
        Install-Packages $py
    }

    # 4 --- verify (rebuild once if broken)
    Step 4 $total 'Verifying packages'
    & $script:VenvPy $SelfCheck --imports-only | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Warn 'some packages are broken - rebuilding the environment from scratch (one time) ...'
        Remove-Item -LiteralPath $script:VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        Install-Packages $py
        & $script:VenvPy $SelfCheck --imports-only | Out-Host
        if ($LASTEXITCODE -ne 0) { Die ('packages still fail to load after a clean install - see the [PACKAGES] lines above and {0}' -f $LogFile) }
    }
    Set-Content -LiteralPath $stampFile -Value $reqHash -Encoding ASCII
    Ok 'all packages load'

    # 5 --- camera + graphics
    Step 5 $total 'Camera, MediaPipe model and graphics'
    $cameraArgs = @()
    $camIdx = $null
    for ($i = 0; $i -lt $AppArgs.Count - 1; $i++) { if ($AppArgs[$i] -eq '--camera') { $camIdx = $AppArgs[$i + 1] } }
    if ($camIdx) { $cameraArgs = @('--camera', $camIdx) }
    if ($AppArgs -contains '--no-camera' -or $AppArgs -contains '--demo') {
        $cameraArgs = @('--no-camera')
    } else {
        Confirm-CameraAccess
    }
    & $script:VenvPy $SelfCheck @cameraArgs | Out-Host
    $rc = $LASTEXITCODE
    if ($rc -eq 1) { Die ('the MediaPipe hand model could not load - see {0}' -f $LogFile) }
    if ($rc -eq 2) { Die 'this PC cannot create an OpenGL 4.1 window - update the graphics driver (Windows Update, or intel.com / nvidia.com / amd.com), then run START_WINDOWS.bat again. It will not work inside Remote Desktop or a virtual machine.' }
    if ($rc -eq 3) {
        Warn 'No working camera was found, so the game will start in KEYBOARD mode.'
        Warn 'To use hand tracking: close apps that use the camera (Zoom, Teams, Camera app, browser tabs),'
        Warn 'check the laptop camera shutter / Fn camera key, allow camera access in Windows Settings,'
        Warn 'then start again. A USB camera on another index can be chosen with:  START_WINDOWS.bat --camera 1'
    } elseif ($rc -eq 0) {
        Ok 'camera, hand model and graphics are ready'
    }

    if ($CheckOnly) { Say ''; Say '  --check finished.' 'Green'; return }
    if ($SetupOnly) { Say ''; Say '  Setup finished. Start the game any time with START_WINDOWS.bat' 'Green'; return }

    # 6 --- launch
    Step 6 $total 'Starting TITAN-9'
    Say '      Show your open hand to the camera. Close the TITAN-9 window to quit.'
    Say '      Game too slow?  Start with:  START_WINDOWS.bat --quality medium   (or low)'
    Say ''
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:GLOG_minloglevel = '2'
    $env:OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS = '0'
    Set-Location -LiteralPath $Root
    & $script:VenvPy -m titan @AppArgs
    $script:AppExit = $LASTEXITCODE
    if ($script:AppExit -ne 0) { Warn ("TITAN-9 stopped with an error (code {0}) - see the messages above." -f $script:AppExit) }
}

# ---- run ---------------------------------------------------------------------
try { Start-Transcript -Path $LogFile -Force | Out-Null } catch { }
try {
    Invoke-Main
} catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    $script:AppExit = 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
exit $script:AppExit
