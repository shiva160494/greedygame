param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$CodexPython = "C:\Users\Shiyam\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $CodexPython) {
    & $CodexPython -m greedygame.cli @Command
    exit $LASTEXITCODE
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source -m greedygame.cli @Command
    exit $LASTEXITCODE
}

Write-Host "Python was not found. Install Python from https://www.python.org/downloads/ and select 'Add python.exe to PATH'." -ForegroundColor Red
exit 1
