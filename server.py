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
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
SCREENSHOT_SAMPLE_STEP = 0.5
SCREENSHOT_STABLE_DIFF = 0.002
SCREENSHOT_PIXEL_DELTA = 0.08
SCREENSHOT_SETTLE_LIMIT = 3.0

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


def _extract_screenshots(path: Path, target_dir: Path, threshold: float, min_interval: float,
                         max_count: int, on_progress) -> list[tuple[float, str]]:
    """Сохраняет кадры при смене картинки; возвращает список (время начала смены, имя файла).

    Кадр-кандидат сохраняется, только когда картинка перестала меняться (или через
    SCREENSHOT_SETTLE_LIMIT секунд), чтобы не ловить середину перехода.
    """
    import av
    import numpy as np

    def thumb(frame):
        return frame.reformat(width=320, height=180, format="gray").to_ndarray().astype(np.float32) / 255.0

    def diff(a, b):
        return float(np.mean(np.abs(a - b) > SCREENSHOT_PIXEL_DELTA))

    shots: list[tuple[float, str]] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    def save(frame, at: float):
        name = "{:02d}-{:02d}-{:02d}.jpg".format(int(at) // 3600, int(at) % 3600 // 60, int(at) % 60)
        frame.to_image().save(target_dir / name, quality=90)
        shots.append((at, name))
        on_progress(len(shots))

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            return shots
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        next_sample = 0.0
        saved_thumb = prev_thumb = None
        saved_at = -min_interval
        pending_at: float | None = None
        for frame in container.decode(stream):
            if frame.time is None or frame.time < next_sample:
                continue
            next_sample = frame.time + SCREENSHOT_SAMPLE_STEP
            current = thumb(frame)
            if saved_thumb is None:
                save(frame, frame.time)
                saved_thumb, saved_at = current, frame.time
            elif pending_at is None:
                if frame.time - saved_at >= min_interval and diff(current, saved_thumb) > threshold:
                    pending_at = frame.time
            elif (diff(current, prev_thumb) < SCREENSHOT_STABLE_DIFF
                  or frame.time - pending_at >= SCREENSHOT_SETTLE_LIMIT):
                if diff(current, saved_thumb) > threshold:
                    save(frame, pending_at)
                    saved_thumb, saved_at = current, frame.time
                pending_at = None
            prev_thumb = current
            if len(shots) >= max_count:
                break
    return shots


def _update(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    source = Path(job["file_path"])
    out_file = Path(job["output_file"])
    shots: list[tuple[float, str]] = []
    shots_error: list[str] = []
    shots_thread = None
    if job["screenshots"]:
        def collect():
            try:
                shots.extend(_extract_screenshots(
                    source, out_file.parent / "screenshots", job["scene_threshold"],
                    job["min_interval"], job["max_screenshots"],
                    lambda n: _update(job_id, screenshots=n)))
            except Exception as exc:  # noqa: BLE001
                logger.error("screenshots failed: %s", traceback.format_exc())
                shots_error.append(f"{type(exc).__name__}: {exc}")

        shots_thread = threading.Thread(target=collect, name=f"screenshots-{job_id}", daemon=True)
        shots_thread.start()

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

    lines: list[tuple[float, str]] = []
    for segment in segments:
        lines.append((segment.start, f"[{_fmt(segment.start)}] {segment.text.strip()}"))
        _update(job_id, segments=len(lines), position=segment.end)

    if shots_thread is not None:
        _update(job_id, status="extracting_screenshots")
        shots_thread.join()

    header = (
        "# Транскрипт\n\n"
        f"Запись: {source.name}, длительность {_fmt(info.duration)}.\n"
        f"Распознавание: faster-whisper {job['model']} ({device}), язык {info.language} "
        f"({info.language_probability:.0%}).\n"
        "Спикеры не разделены: диаризация недоступна.\n"
    )
    if job["screenshots"]:
        header += (f"Скриншоты: {len(shots)} при смене картинки, каталог screenshots/.\n"
                   if not shots_error else f"Скриншоты не извлечены: {shots_error[0]}\n")
    body: list[tuple[float, int, str]] = [(at, 0, line) for at, line in lines]
    body += [(at, -1, f"\n![{_fmt(at)}](screenshots/{name})\n") for at, name in shots]
    body.sort(key=lambda item: (item[0], item[1]))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(header + "\n" + "\n".join(item[2] for item in body) + "\n", encoding="utf-8")
    _update(job_id, status="done", finished_at=time.time(), screenshots=len(shots),
            screenshots_error=shots_error[0] if shots_error else None)


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
    screenshots: Annotated[bool, Field(description="Сохранять скриншоты при смене картинки (только для видео)")] = True,
    scene_threshold: Annotated[float, Field(description="Порог смены картинки: доля изменившихся пикселей экрана 0–1; меньше — больше скриншотов", gt=0, le=1)] = 0.02,
    min_interval: Annotated[float, Field(description="Минимум секунд между скриншотами", ge=1)] = 5.0,
    max_screenshots: Annotated[int, Field(description="Максимум скриншотов на файл", ge=1, le=2000)] = 300,
) -> str:
    """Ставит файл в очередь локальной расшифровки и возвращает job_id.

    Результат — transcript.md с метками времени; для видео — скриншоты при смене картинки
    в каталоге screenshots/ со ссылками в тексте. Задачи выполняются по одной.
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
            "screenshots": screenshots and source.suffix.lower() in VIDEO_EXTENSIONS,
            "scene_threshold": scene_threshold,
            "min_interval": min_interval,
            "max_screenshots": max_screenshots,
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
    """Статус задачи: queued, decoding_audio, loading_model, transcribing, extracting_screenshots, done, error.

    Также прогресс, число скриншотов и путь к результату.
    """
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
