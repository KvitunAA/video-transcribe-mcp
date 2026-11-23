# Быстрый старт video-transcribe-mcp: зависимости, модель, проверка GPU, фрагмент mcp.json.
param(
    [string]$Python = "python",
    [string]$Model = "large-v3-turbo"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Write-Host "[1/4] Python" -ForegroundColor Cyan
$pyExe = & $Python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "Python не найден: $Python" }
Write-Host "  $pyExe"

Write-Host "[2/4] Зависимости (faster-whisper, fastmcp, CUDA cuBLAS/cuDNN)" -ForegroundColor Cyan
& $pyExe -m pip install --disable-pip-version-check -r (Join-Path $here "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install завершился с ошибкой" }

Write-Host "[3/4] Модель $Model" -ForegroundColor Cyan
$download = @"
import time
from faster_whisper.utils import _MODELS
from huggingface_hub import snapshot_download
repo = _MODELS.get('$Model', '$Model')
for attempt in range(1, 11):
    try:
        print('  ', snapshot_download(repo, allow_patterns=['config.json', 'preprocessor_config.json', 'model.bin', 'tokenizer.json', 'vocabulary*']))
        break
    except Exception as exc:
        print(f'  попытка {attempt}: {type(exc).__name__}: {exc}')
        time.sleep(5)
else:
    raise SystemExit('Не удалось скачать модель')
"@
$download | & $pyExe -
if ($LASTEXITCODE -ne 0) { throw "Загрузка модели не удалась" }

Write-Host "[4/4] Проверка загрузки модели" -ForegroundColor Cyan
$check = @"
import sys
sys.path.insert(0, r'$here')
import server
model, device = server._load_model('$Model')
print('  устройство:', device)
"@
$check | & $pyExe -
if ($LASTEXITCODE -ne 0) { throw "Модель не загрузилась" }

$server = Join-Path $here "server.py"
$snippet = [ordered]@{
    "video-transcribe" = [ordered]@{
        command     = $pyExe
        args        = @($server)
        env         = [ordered]@{ PYTHONUNBUFFERED = "1"; WHISPER_MODEL = $Model }
        description = "Локальная расшифровка видео/аудио: faster-whisper на GPU, transcript.md с таймкодами"
    }
}
Write-Host "`nГотово. Добавьте в mcpServers файла ~/.cursor/mcp.json:" -ForegroundColor Green
$snippet | ConvertTo-Json -Depth 5
