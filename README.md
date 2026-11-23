# video-transcribe-mcp

MCP-сервер для локальной расшифровки видео и аудио в текст. Работает полностью на вашем компьютере: запись никуда не отправляется, API-ключи не нужны.

- Распознавание — [faster-whisper](https://github.com/SYSTRAN/faster-whisper), модель **large-v3-turbo** (около 1,6 ГБ).
- Видеокарта NVIDIA (CUDA, float16); если не получилось — int8 на видеокарте, затем процессор.
- Звук декодируется прямо из файла через PyAV, ffmpeg не нужен.
- Результат — `transcript.md` с таймкодами `[MM:SS]` или `[HH:MM:SS]`.

Пример: 19,5 минуты записи встречи на RTX 5070 Ti расшифровываются примерно за 40 секунд.

## Быстрый старт

Требуется Windows, Python 3.10+ и, для работы на GPU, видеокарта NVIDIA со свежим драйвером.

```powershell
git clone https://github.com/KvitunAA/video-transcribe-mcp.git
cd video-transcribe-mcp
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

`install.ps1` скачивает всё необходимое:

1. Python-пакеты из `requirements.txt`: `faster-whisper`, `fastmcp`, CUDA-библиотеки `nvidia-cublas-cu12` и `nvidia-cudnn-cu12` (вместе около 1,2 ГБ).
2. Модель Whisper в кэш HuggingFace (`~/.cache/huggingface`), с повторами при обрывах.
3. Проверяет загрузку модели и показывает устройство (`cuda/float16` или `cpu/int8`).
4. Выводит готовый фрагмент для `mcp.json` с путями вашей машины.

Параметры: `-Python` — путь к интерпретатору (по умолчанию `python`), `-Model` — другая модель faster-whisper, например `large-v3`, `medium`, `small`.

Скопируйте выведенный фрагмент в `mcpServers` файла `%USERPROFILE%\.cursor\mcp.json` и перезапустите MCP-серверы в Cursor.

## Инструменты

| Инструмент | Назначение |
|---|---|
| `start_transcription` | Ставит файл в очередь, сразу возвращает `job_id` и путь к будущему `transcript.md` |
| `get_transcription_status` | Статус, процент готовности, путь к результату; с `include_text=true` — сам текст |

Расшифровка длинной записи может не уложиться в таймаут одного вызова MCP, поэтому запуск и получение результата разделены. Параметр `wait_seconds` (до 600) позволяет дождаться завершения в одном вызове.

Параметры `start_transcription`:

- `file_path` — путь к файлу: mp4, mkv, webm, avi, mov, mp3, wav, ogg, m4a, flac, aac, wma;
- `output_dir` — каталог результата, по умолчанию `<папка файла>\Transcript\<имя файла>\`;
- `language` — код языка, по умолчанию `ru`; пусто — автоопределение;
- `model` — модель faster-whisper, по умолчанию из переменной `WHISPER_MODEL` или `large-v3-turbo`;
- `initial_prompt` — подсказка с терминами и именами для лучшего распознавания;
- `beam_size` — ширина луча декодирования, 1–10, по умолчанию 5.

Статусы задачи: `queued`, `decoding_audio`, `loading_model`, `transcribing`, `done`, `error`.

## Ограничения

- Спикеры не разделяются.
- Задачи выполняются по одной; очередь хранится в памяти и пропадает при перезапуске сервера, готовые файлы остаются.

## Если что-то не работает

- **Модель не скачивается или загрузка стоит на нуле.** Сервер и `install.ps1` отключают механизм Xet (`HF_HUB_DISABLE_XET=1`), с которым загрузка с HuggingFace может зависать. Если обрывы продолжаются, перезапустите `install.ps1`: уже скачанное сохраняется.
- **Видеокарта не используется (`cpu/int8`).** Проверьте драйвер NVIDIA (`nvidia-smi`) и что пакеты `nvidia-cublas-cu12` и `nvidia-cudnn-cu12` установлены в тот же Python, которым запускается сервер.
