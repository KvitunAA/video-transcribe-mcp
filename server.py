"""MCP-сервер: локальная расшифровка видео и аудио через faster-whisper (GPU/CPU)."""

from __future__ import annotations

import json
import logging
import os
import queue
import site
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Annotated, Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

for _sp in site.getsitepackages() + [site.getusersitepackages()]:
    for _sub in ("cublas", "cudnn"):
        _bin = Path(_sp) / "nvidia" / _sub / "bin"
        if _bin.is_dir():
            os.add_dll_directory(str(_bin))
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

try:
    from fastmcp import FastMCP
    from pydantic import Field
except ImportError as exc:
    raise SystemExit("Установите зависимости: pip install -r requirements.txt") from exc

APP_NAME = "video-transcribe"
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".avi", ".mov",
    ".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma",
}

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(APP_NAME)
mcp = FastMCP(APP_NAME)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_models: dict[str, tuple[Any, str]] = {}


def _load_model(name: str):
    """Загружает модель один раз: CUDA float16 → CUDA int8_float16 → CPU int8."""
    if name in _models:
        return _models[name]
    from faster_whisper import WhisperModel

    last_error: Exception | None = None
    for device, compute_type in (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            logger.info("load model=%s device=%s compute=%s", name, device, compute_type)
            model = WhisperModel(name, device=device, compute_type=compute_type)
            _models[name] = (model, f"{device}/{compute_type}")
            return _models[name]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("load failed device=%s: %s", device, exc)
    raise RuntimeError(f"Не удалось загрузить модель {name}: {last_error}")


def _decode_audio(path: Path):
    """Декодирует звуковую дорожку в моно 16 кГц float32.

    Встроенный decode_audio faster-whisper несовместим с PyAV 15+ (нет metadata_errors).
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"В файле нет звуковой дорожки: {path}")
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"Звуковая дорожка пуста: {path}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _fmt(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _update(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    source = Path(job["file_path"])
    out_file = Path(job["output_file"])
    _update(job_id, status="decoding_audio", started_at=time.time())
    audio = _decode_audio(source)
    _update(job_id, status="loading_model")
    model, device = _load_model(job["model"])
    _update(job_id, status="transcribing", device=device)

    segments, info = model.transcribe(
        audio,
        language=job["language"],
        vad_filter=True,
        beam_size=job["beam_size"],
        initial_prompt=job["initial_prompt"],
        condition_on_previous_text=True,
    )
    _update(job_id, duration=info.duration, language=info.language)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Транскрипт\n\n"
        f"Запись: {source.name}, длительность {_fmt(info.duration)}.\n"
        f"Распознавание: faster-whisper {job['model']} ({device}), язык {info.language} "
        f"({info.language_probability:.0%}).\n"
        "Спикеры не разделены: диаризация недоступна.\n\n"
    )
    count = 0
    with out_file.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for segment in segments:
            handle.write(f"[{_fmt(segment.start)}] {segment.text.strip()}\n")
            handle.flush()
            count += 1
            _update(job_id, segments=count, position=segment.end)
    _update(job_id, status="done", finished_at=time.time())


def _worker() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("job %s failed: %s", job_id, traceback.format_exc())
            _update(job_id, status="error", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
        finally:
            _queue.task_done()


threading.Thread(target=_worker, name="transcribe-worker", daemon=True).start()


def _public(job: dict[str, Any], include_text: bool) -> dict[str, Any]:
    result = {k: v for k, v in job.items() if k not in ("initial_prompt",)}
    if job.get("duration") and job.get("position") is not None:
        result["progress_percent"] = round(min(100.0, job["position"] / job["duration"] * 100), 1)
    for key in ("started_at", "finished_at", "created_at"):
        if result.get(key):
            result[key] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result[key]))
    if include_text and job["status"] == "done":
        result["transcript"] = Path(job["output_file"]).read_text(encoding="utf-8")
    return result


@mcp.tool()
def start_transcription(
    file_path: Annotated[str, Field(description="Абсолютный путь к видео или аудио (mp4, mkv, webm, avi, mov, mp3, wav, ogg, m4a, flac, aac, wma)")],
    output_dir: Annotated[str | None, Field(description="Каталог результата; по умолчанию <папка файла>/Transcript/<имя файла>/")] = None,
    language: Annotated[str | None, Field(description="Код языка (ru, en…); пусто — автоопределение")] = "ru",
    model: Annotated[str | None, Field(description="Модель faster-whisper; по умолчанию large-v3-turbo")] = None,
    initial_prompt: Annotated[str | None, Field(description="Подсказка с терминами и именами для лучшего распознавания")] = None,
    beam_size: Annotated[int, Field(description="Ширина луча декодирования", ge=1, le=10)] = 5,
) -> str:
    """Ставит файл в очередь локальной расшифровки и возвращает job_id.

    Результат — transcript.md с метками времени. Задачи выполняются по одной.
    Статус и текст — через get_transcription_status.
    """
    source = Path(file_path)
    if not source.is_file():
        raise ValueError(f"Файл не найден: {file_path}")
    if source.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {source.suffix}")
    target_dir = Path(output_dir) if output_dir else source.parent / "Transcript" / source.stem

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "file_path": str(source),
            "output_file": str(target_dir / "transcript.md"),
            "model": model or DEFAULT_MODEL,
            "language": language or None,
            "initial_prompt": initial_prompt,
            "beam_size": beam_size,
            "segments": 0,
            "created_at": time.time(),
        }
        ahead = sum(1 for j in _jobs.values() if j["status"] not in ("done", "error")) - 1
    _queue.put(job_id)
    return json.dumps({"job_id": job_id, "queued_ahead": ahead, "output_file": _jobs[job_id]["output_file"]},
                      ensure_ascii=False, indent=2)


@mcp.tool()
def get_transcription_status(
    job_id: Annotated[str | None, Field(description="Идентификатор задачи; пусто — список всех задач")] = None,
    wait_seconds: Annotated[int, Field(description="Сколько секунд ждать завершения перед ответом", ge=0, le=600)] = 0,
    include_text: Annotated[bool, Field(description="Вернуть текст транскрипта, если задача завершена")] = False,
) -> str:
    """Статус задачи: queued, decoding_audio, loading_model, transcribing, done, error; прогресс и путь к результату."""
    if not job_id:
        with _jobs_lock:
            items = [_public(j, False) for j in _jobs.values()]
        return json.dumps(items, ensure_ascii=False, indent=2)
    if job_id not in _jobs:
        raise ValueError(f"Задача не найдена: {job_id}")
    deadline = time.time() + wait_seconds
    while _jobs[job_id]["status"] not in ("done", "error") and time.time() < deadline:
        time.sleep(2)
    with _jobs_lock:
        return json.dumps(_public(_jobs[job_id], include_text), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
