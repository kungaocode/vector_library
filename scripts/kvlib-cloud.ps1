# Full cloud pipeline: Qwen annotation + Qwen embeddings + local PostgreSQL.
# Usage: .\scripts\kvlib-cloud.ps1 ingest
#        .\scripts\kvlib-cloud.ps1 search "query"
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$python = 'F:\vector_library\pipeline-venv\Scripts\python.exe'
$configPath = Join-Path $projectRoot '.knowledge-vector\cloud.toml'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $configPath)) { throw "Configuration not found: $configPath" }
$cliArguments = @($args)
if (-not $cliArguments.Count) { $cliArguments = @('stats') }
$previousEncoding = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    & $python -m knowledge_vector @cliArguments --config $configPath
    if ($LASTEXITCODE -ne 0) { throw "kvlib failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONUTF8 = $previousEncoding
}
