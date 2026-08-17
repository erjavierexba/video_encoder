#!/usr/bin/env python3
"""
UVHide Decoder
==============

Compatible con UVHide Encoder 5x.

MODOS

1) Reproductor (por defecto)
   uvhide-decoder video.mp4

   - Verifica primero el mensaje completo mediante SHA-256.
   - Reproduce el vídeo.
   - Va revelando las letras en forma de subtítulo.
   - Hace sonar un pequeño timbre por cada nueva letra.
   - El subtítulo reduce automáticamente la fuente para ocupar como máximo
     dos líneas.
   - Controles superiores:
       * rebobinar (mantener pulsado)
       * 0.5x
       * play/pausa
       * 2x
   - El timbre queda silenciado mientras hay un modificador activo
     (rebobinado, 0.5x o 2x).

2) Solo texto
   uvhide-decoder video.mp4 --text

   Imprime exclusivamente el texto final en stdout si el SHA-256 es válido.

DEPENDENCIAS DE DESARROLLO

   numpy
   ffmpeg / ffprobe

Para el modo gráfico:
   PySide6

La versión empaquetada puede incluir sus dependencias y ffmpeg/ffprobe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError:
    print(
        "ERROR: falta NumPy.\n"
        "En desarrollo instala:\n"
        "  python -m pip install numpy",
        file=sys.stderr,
    )
    raise SystemExit(2)


# Debe ser idéntico al encoder 5x.
DATA_COORDS = tuple(
    (x, y)
    for y in (0.12, 0.36, 0.64, 0.88)
    for x in (0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92)
)

HASH_WORDS = 8
DEFAULT_DELTA0 = 3
DEFAULT_DELTA1 = 7


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frame_count: int


@dataclass(frozen=True)
class CharacterEvent:
    frame_index: int
    time_ms: int
    character: str
    confidence: float


@dataclass(frozen=True)
class DecodeResult:
    text: str
    sha256_hex: str
    hash_valid: bool
    events: tuple[CharacterEvent, ...]
    length: int
    source_frames: int
    total_frames: int
    source_duration_ms: int
    mean_confidence: float


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def bundled_base_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def find_binary(name: str) -> str:
    base = bundled_base_dir()

    candidates = (
        base / name,
        base / f"{name}.exe",
        base / "bin" / name,
        base / "bin" / f"{name}.exe",
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    found_exe = shutil.which(f"{name}.exe")
    if found_exe:
        return found_exe

    die(
        f"No encuentro '{name}'. En desarrollo instálalo en PATH; "
        "en el ejecutable final debe ir incluido."
    )


FFMPEG: str | None = None
FFPROBE: str | None = None


def init_external_tools() -> None:
    global FFMPEG, FFPROBE
    FFMPEG = find_binary("ffmpeg")
    FFPROBE = find_binary("ffprobe")


def run_json(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        die(exc.stderr.strip() or f"Falló: {' '.join(cmd)}")

    return json.loads(proc.stdout)


def probe_video(path: Path) -> VideoInfo:
    assert FFPROBE is not None

    info = run_json([
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json",
        str(path),
    ])

    streams = info.get("streams", [])
    if not streams:
        die("El archivo no contiene una pista de vídeo.")

    stream = streams[0]

    width = int(stream["width"])
    height = int(stream["height"])

    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not fps_text or fps_text == "0/0":
        die("No he podido determinar los FPS.")

    fps = Fraction(fps_text)
    if fps <= 0:
        die("FPS inválidos.")

    nb_frames = stream.get("nb_frames")

    if nb_frames and str(nb_frames).isdigit():
        frame_count = int(nb_frames)
    else:
        counted = run_json([
            FFPROBE,
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "json",
            str(path),
        ])
        try:
            frame_count = int(counted["streams"][0]["nb_read_frames"])
        except (KeyError, IndexError, TypeError, ValueError):
            die("No he podido determinar el número exacto de frames.")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def scaled_coordinates(width: int, height: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []

    for nx, ny in DATA_COORDS:
        x = int(round(nx * (width - 1)))
        y = int(round(ny * (height - 1)))
        coords.append((x, y))

    if len(set(coords)) != 32:
        die("La resolución produce coordenadas de datos solapadas.")

    return coords


def sample_spacing(width: int, height: int) -> int:
    return max(3, int(round(min(width, height) / 270.0)))


def sample_positions(
    x: int,
    y: int,
    spacing: int,
) -> tuple[tuple[int, int], ...]:
    return (
        (x, y - spacing),
        (x - spacing, y),
        (x, y),
        (x + spacing, y),
        (x, y + spacing),
    )


def validate_sample_positions(
    width: int,
    height: int,
    logical_coords: list[tuple[int, int]],
    control_xy: tuple[int, int],
    spacing: int,
) -> None:
    physical: list[tuple[int, int]] = []

    for logical_xy in [*logical_coords, control_xy]:
        for x, y in sample_positions(*logical_xy, spacing):
            if not (1 <= x < width - 1 and 1 <= y < height - 1):
                die("La resolución no permite leer las muestras de forma segura.")
            physical.append((x, y))

    if len(set(physical)) != len(physical):
        die("Hay muestras físicas solapadas.")


def neighbor_mean_rgb(
    frame: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    block = frame[y - 1:y + 2, x - 1:x + 2].astype(np.int16)
    total = block.sum(axis=(0, 1)) - block[1, 1]
    return total / 8.0


def read_physical_sample_score(
    frame: np.ndarray,
    x: int,
    y: int,
) -> float:
    """
    Magnitud de la anomalía local del píxel respecto a sus 8 vecinos.

    El encoder escribe aproximadamente delta0 o delta1 en cada canal.
    Usamos la media de las diferencias absolutas RGB.
    """
    mean_rgb = neighbor_mean_rgb(frame, x, y)
    center = frame[y, x].astype(np.float64)
    return float(np.mean(np.abs(center - mean_rgb)))


def read_logical_bit(
    frame: np.ndarray,
    x: int,
    y: int,
    spacing: int,
    threshold: float,
) -> tuple[int, float]:
    """
    Lee las 5 muestras físicas y decide por mayoría.

    Devuelve:
        bit
        confidence 0..1

    La confianza combina:
      - acuerdo de voto (3/5, 4/5, 5/5)
      - separación media respecto al umbral.
    """
    raw_scores: list[float] = []
    votes: list[int] = []

    for sx, sy in sample_positions(x, y, spacing):
        score = read_physical_sample_score(frame, sx, sy)
        raw_scores.append(score)
        votes.append(1 if score >= threshold else 0)

    ones = sum(votes)
    bit = 1 if ones >= 3 else 0

    agreeing = ones if bit else 5 - ones
    vote_conf = agreeing / 5.0

    # Cuanto más lejos del umbral, más clara es la separación.
    # La normalización es deliberadamente conservadora.
    mean_margin = float(np.mean([abs(score - threshold) for score in raw_scores]))
    margin_conf = min(1.0, mean_margin / max(1.0, threshold))

    confidence = 0.75 * vote_conf + 0.25 * margin_conf
    return bit, confidence


def bits_to_word(bits: Iterable[int]) -> int:
    word = 0
    for bit in bits:
        word = (word << 1) | (1 if bit else 0)
    return word


def read_word(
    frame: np.ndarray,
    coords: list[tuple[int, int]],
    control_xy: tuple[int, int],
    spacing: int,
    threshold: float,
) -> tuple[int, int, float]:
    bits: list[int] = []
    confidences: list[float] = []

    for x, y in coords:
        bit, confidence = read_logical_bit(
            frame,
            x,
            y,
            spacing,
            threshold,
        )
        bits.append(bit)
        confidences.append(confidence)

    control, control_conf = read_logical_bit(
        frame,
        control_xy[0],
        control_xy[1],
        spacing,
        threshold,
    )

    confidence = float(np.mean([*confidences, control_conf]))

    return bits_to_word(bits), control, confidence


def expected_char_index(
    frame_index: int,
    source_frames: int,
    char_count: int,
) -> int:
    """
    Misma distribución que el encoder.

    frame_index debe ser 1..source_frames-1.
    """
    available = source_frames - 1
    return min(
        ((frame_index - 1) * char_count) // available,
        char_count - 1,
    )


def char_start_frame(
    char_index: int,
    source_frames: int,
    char_count: int,
) -> int:
    """
    Primer frame en el que aparece char_index.

    Usa ceil(char_index * available / char_count) sin floats.
    """
    available = source_frames - 1
    numerator = char_index * available
    t = (numerator + char_count - 1) // char_count
    return 1 + t


def frame_to_ms(frame_index: int, fps: Fraction) -> int:
    seconds = Fraction(frame_index, 1) / fps
    return int(round(float(seconds) * 1000.0))


def aggregate_words_bitwise(
    words: list[tuple[int, float]],
) -> tuple[int, float]:
    """
    Combina todos los frames pertenecientes al mismo carácter a nivel de bit.

    Esto es más robusto que escoger la palabra de 32 bits más frecuente:
    si distintos frames sufren errores en bits diferentes, cada bit todavía
    puede recuperarse por mayoría temporal.

    Si el carácter dura un solo frame, conserva el resultado de las 5
    muestras espaciales de ese frame.
    """
    if not words:
        raise ValueError("No hay muestras para el carácter.")

    final_bits: list[int] = []
    bit_confidences: list[float] = []

    for shift in range(31, -1, -1):
        weighted_ones = 0.0
        weighted_zeroes = 0.0

        for word, frame_confidence in words:
            bit = (word >> shift) & 1
            weight = max(0.01, frame_confidence)

            if bit:
                weighted_ones += weight
            else:
                weighted_zeroes += weight

        total = weighted_ones + weighted_zeroes
        bit = 1 if weighted_ones >= weighted_zeroes else 0
        final_bits.append(bit)

        winning = max(weighted_ones, weighted_zeroes)
        bit_confidences.append(
            winning / total if total else 0.0
        )

    word = bits_to_word(final_bits)
    confidence = float(np.mean(bit_confidences))
    return word, confidence


def validate_unicode_codepoint(value: int) -> bool:
    if value > 0x10FFFF:
        return False
    if 0xD800 <= value <= 0xDFFF:
        return False
    return True


def decode_video(
    video_path: Path,
    delta0: int = DEFAULT_DELTA0,
    delta1: int = DEFAULT_DELTA1,
    progress_callback=None,
) -> DecodeResult:
    global FFMPEG, FFPROBE

    init_external_tools()
    assert FFMPEG is not None
    assert FFPROBE is not None

    if not video_path.exists():
        die(f"No existe: {video_path}")

    if not (0 <= delta0 < delta1 <= 64):
        die("Debe cumplirse 0 <= delta0 < delta1 <= 64.")

    info = probe_video(video_path)

    if info.frame_count <= HASH_WORDS + 1:
        die("El vídeo es demasiado corto para contener un mensaje UVHide.")

    source_frames = info.frame_count - HASH_WORDS
    coords = scaled_coordinates(info.width, info.height)
    control_xy = (info.width // 2, info.height // 2)
    spacing = sample_spacing(info.width, info.height)

    validate_sample_positions(
        info.width,
        info.height,
        coords,
        control_xy,
        spacing,
    )

    threshold = (delta0 + delta1) / 2.0
    frame_bytes = info.width * info.height * 3

    decoder = subprocess.Popen(
        [
            FFMPEG,
            "-v", "error",
            "-i", str(video_path),
            "-map", "0:v:0",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-fps_mode", "passthrough",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if decoder.stdout is None:
        die("No se pudo abrir el stream de FFmpeg.")

    declared_length: int | None = None
    char_samples: list[list[tuple[int, float]]] | None = None
    control_matches: list[bool] = []
    hash_words: list[int] = []
    hash_confidences: list[float] = []

    processed = 0

    try:
        while processed < info.frame_count:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                (info.height, info.width, 3)
            )

            word, control, confidence = read_word(
                frame,
                coords,
                control_xy,
                spacing,
                threshold,
            )

            if processed == 0:
                declared_length = word

                # Evita interpretar ruido aleatorio como una longitud absurda.
                max_chars = source_frames - 1
                if declared_length <= 0 or declared_length > max_chars:
                    decoder.kill()
                    die(
                        "No parece un vídeo UVHide válido: "
                        f"longitud detectada={declared_length}, "
                        f"máximo posible={max_chars}."
                    )

                char_samples = [
                    []
                    for _ in range(declared_length)
                ]

                control_matches.append(control == 0)

            elif processed < source_frames:
                assert declared_length is not None
                assert char_samples is not None

                char_index = expected_char_index(
                    processed,
                    source_frames,
                    declared_length,
                )

                char_samples[char_index].append(
                    (word, confidence)
                )

                expected_control = (char_index + 1) & 1
                control_matches.append(
                    control == expected_control
                )

            else:
                hash_index = processed - source_frames
                if hash_index < HASH_WORDS:
                    hash_words.append(word)
                    hash_confidences.append(confidence)

                    assert declared_length is not None
                    expected_control = (
                        declared_length
                        + hash_index
                        + 1
                    ) & 1
                    control_matches.append(
                        control == expected_control
                    )

            processed += 1

            if progress_callback:
                progress_callback(processed, info.frame_count)

    finally:
        if decoder.stdout:
            decoder.stdout.close()

    decoder_rc = decoder.wait()

    decoder_err = (
        decoder.stderr.read().decode("utf-8", "replace")
        if decoder.stderr
        else ""
    )

    if decoder_rc != 0:
        die(f"FFmpeg falló decodificando:\n{decoder_err.strip()}")

    if processed != info.frame_count:
        die(
            "Número inesperado de frames durante la lectura: "
            f"{processed}/{info.frame_count}"
        )

    assert declared_length is not None
    assert char_samples is not None

    if len(hash_words) != HASH_WORDS:
        die("No se pudieron recuperar los 8 frames del SHA-256.")

    decoded_chars: list[str] = []
    events: list[CharacterEvent] = []
    char_confidences: list[float] = []

    for index, samples in enumerate(char_samples):
        if not samples:
            die(f"No hay frames disponibles para el carácter {index}.")

        word, confidence = aggregate_words_bitwise(samples)

        if not validate_unicode_codepoint(word):
            die(
                f"Carácter Unicode inválido detectado en posición {index}: "
                f"U+{word:08X}"
            )

        char = chr(word)
        decoded_chars.append(char)
        char_confidences.append(confidence)

        start_frame = char_start_frame(
            index,
            source_frames,
            declared_length,
        )

        events.append(
            CharacterEvent(
                frame_index=start_frame,
                time_ms=frame_to_ms(start_frame, info.fps),
                character=char,
                confidence=confidence,
            )
        )

    text = "".join(decoded_chars)

    recovered_digest = b"".join(
        word.to_bytes(4, "big")
        for word in hash_words
    )

    calculated_digest = hashlib.sha256(
        declared_length.to_bytes(4, "big")
        + text.encode("utf-8")
    ).digest()

    hash_valid = recovered_digest == calculated_digest

    all_conf = [
        *char_confidences,
        *hash_confidences,
    ]

    if control_matches:
        control_conf = sum(control_matches) / len(control_matches)
        all_conf.append(control_conf)

    mean_confidence = (
        float(np.mean(all_conf))
        if all_conf
        else 0.0
    )

    return DecodeResult(
        text=text,
        sha256_hex=recovered_digest.hex(),
        hash_valid=hash_valid,
        events=tuple(events),
        length=declared_length,
        source_frames=source_frames,
        total_frames=info.frame_count,
        source_duration_ms=frame_to_ms(source_frames, info.fps),
        mean_confidence=mean_confidence,
    )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def create_chime_wav(path: Path) -> None:
    """
    Genera un timbre corto sin necesitar un asset externo.
    WAV PCM mono 44.1 kHz.
    """
    sample_rate = 44100
    duration = 0.105
    count = int(sample_rate * duration)

    samples = bytearray()

    for i in range(count):
        t = i / sample_rate

        # Ataque rápido y caída suave.
        attack = min(1.0, t / 0.008)
        decay = math.exp(-24.0 * t)
        envelope = attack * decay

        value = (
            math.sin(2.0 * math.pi * 880.0 * t)
            + 0.42 * math.sin(2.0 * math.pi * 1320.0 * t)
        )

        pcm = int(max(-1.0, min(1.0, value * envelope * 0.42)) * 32767)
        samples += struct.pack("<h", pcm)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(samples))


def launch_player(video_path: Path, result: DecodeResult) -> int:
    # El backend FFmpeg de Qt prueba CUDA automáticamente en algunos equipos
    # aunque la GPU no tenga las capacidades necesarias. El vídeo que maneja
    # UVHide no necesita aceleración hardware y la decodificación por software
    # evita esos avisos y los fallos de inicialización asociados.
    os.environ["QT_FFMPEG_DECODING_HW_DEVICE_TYPES"] = ""

    def install_multimedia_stderr_filter():
        """
        Qt 6 puede pedir primero el formato CUDA al decoder software y FFmpeg
        escribe avisos engañosos antes de continuar correctamente por CPU.
        Filtramos solo esas líneas conocidas y conservamos el resto de stderr.
        """
        if os.name != "posix":
            return lambda: None

        ignored = (
            b"No HW decoder found",
            b"Invalid setup for format cuda",
            b"Hardware is lacking required capabilities",
            b"Failed setup for format cuda",
        )

        try:
            sys.stderr.flush()
            saved_stderr = os.dup(2)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 2)
            os.close(write_fd)
        except OSError:
            return lambda: None

        def relay() -> None:
            try:
                with os.fdopen(read_fd, "rb", buffering=0) as stream:
                    for line in stream:
                        if any(pattern in line for pattern in ignored):
                            continue
                        os.write(saved_stderr, line)
            except OSError:
                pass

        relay_thread = threading.Thread(
            target=relay,
            name="uvhide-stderr-filter",
            daemon=True,
        )
        relay_thread.start()

        def restore() -> None:
            try:
                sys.stderr.flush()
                os.dup2(saved_stderr, 2)
                relay_thread.join(timeout=1.0)
                os.close(saved_stderr)
            except OSError:
                pass

        return restore

    try:
        from PySide6.QtCore import (
            QRect,
            QTimer,
            Qt,
            QUrl,
        )
        from PySide6.QtGui import (
            QColor,
            QFont,
            QFontMetrics,
            QImage,
            QPainter,
        )
        from PySide6.QtMultimedia import (
            QAudioOutput,
            QMediaPlayer,
            QSoundEffect,
            QVideoSink,
        )
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        die(
            "El modo reproductor necesita PySide6.\n"
            "En desarrollo instala:\n"
            "  python -m pip install PySide6\n"
            "El ejecutable distribuido lo llevará incluido."
        )

    class SubtitleLabel(QLabel):
        """
        QLabel que elige automáticamente la fuente más grande que cabe
        en un máximo visual de dos líneas.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("uvhideSubtitle")
            self.setAlignment(
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignVCenter
            )
            self.setWordWrap(True)
            self.setTextFormat(Qt.TextFormat.PlainText)
            self.setStyleSheet(
                """
                QLabel#uvhideSubtitle {
                    color: white;
                    background-color: rgba(0, 0, 0, 128);
                    border-radius: 10px;
                    padding: 8px 16px;
                }
                """
            )
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                True,
            )

        def set_subtitle_text(self, text: str) -> None:
            self.setText(text)
            self._fit_font()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._fit_font()

        def _fits(self, point_size: int) -> bool:
            text = self.text()
            if not text:
                return True

            font = QFont(self.font())
            font.setPointSize(point_size)
            font.setBold(True)

            metrics = QFontMetrics(font)

            usable_width = max(20, self.width() - 36)
            bounds = metrics.boundingRect(
                QRect(0, 0, usable_width, 100000),
                int(
                    Qt.TextFlag.TextWordWrap
                    | Qt.AlignmentFlag.AlignHCenter
                ),
                text,
            )

            max_height = metrics.lineSpacing() * 2 + 4
            return bounds.height() <= max_height

        def _fit_font(self) -> None:
            if self.width() <= 20:
                return

            low = 6
            high = 54
            best = low

            while low <= high:
                mid = (low + high) // 2
                if self._fits(mid):
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1

            font = QFont(self.font())
            font.setPointSize(best)
            font.setBold(True)
            self.setFont(font)

    class VideoOverlay(QWidget):
        def __init__(self):
            super().__init__()
            self.setAttribute(
                Qt.WidgetAttribute.WA_OpaquePaintEvent,
                True,
            )
            self._image = QImage()

            # Pintamos nosotros mismos los frames recibidos de QMediaPlayer.
            # Así el vídeo y el QLabel comparten el mismo QWidget y ninguna
            # superficie nativa puede tapar los subtítulos en Linux.
            self.video_sink = QVideoSink(self)
            self.video_sink.videoFrameChanged.connect(
                self._video_frame_changed
            )

            self.subtitle = SubtitleLabel(self)
            self.subtitle.hide()

        def _video_frame_changed(self, frame) -> None:
            image = frame.toImage()
            if not image.isNull():
                self._image = image
                self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0))

            if not self._image.isNull():
                image_size = self._image.size()
                image_size.scale(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                target = QRect(
                    (self.width() - image_size.width()) // 2,
                    (self.height() - image_size.height()) // 2,
                    image_size.width(),
                    image_size.height(),
                )
                painter.drawImage(target, self._image)

            painter.end()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._layout_subtitle()

        def _layout_subtitle(self):
            margin_x = max(20, int(self.width() * 0.06))
            width = max(100, self.width() - 2 * margin_x)

            # Reserva suficiente altura para dos líneas incluso con fuente grande.
            box_height = max(80, int(self.height() * 0.24))
            bottom_margin = max(18, int(self.height() * 0.055))

            self.subtitle.setGeometry(
                margin_x,
                max(0, self.height() - box_height - bottom_margin),
                width,
                box_height,
            )

        def set_text(self, text: str):
            if text:
                self._layout_subtitle()
                self.subtitle.show()
                self.subtitle.set_subtitle_text(text)
                self.subtitle.raise_()
                self.subtitle.update()
            else:
                self.subtitle.hide()
                self.subtitle.setText("")

    class MainWindow(QMainWindow):
        NORMAL_RATE = 1.0
        SLOW_RATE = 0.5
        FAST_RATE = 2.0

        def __init__(self):
            super().__init__()

            self.setWindowTitle("UVHide Decoder")
            self.resize(1100, 720)

            self.events = result.events
            self.full_text = result.text
            self.visible_count = 0
            # Los últimos 8 frames pertenecen al SHA-256 y no forman parte
            # del vídeo narrativo que debe ver el jugador.
            self.visual_end_ms = max(0, result.source_duration_ms - 1)

            self.player = QMediaPlayer(self)
            self.audio = QAudioOutput(self)
            self.player.setAudioOutput(self.audio)
            self.audio.setVolume(1.0)

            self.video_overlay = VideoOverlay()
            self.player.setVideoSink(self.video_overlay.video_sink)

            self.chime_tmp = Path(
                tempfile.gettempdir()
            ) / f"uvhide_chime_{os.getpid()}.wav"

            create_chime_wav(self.chime_tmp)

            self.chime = QSoundEffect(self)
            self.chime.setSource(
                QUrl.fromLocalFile(str(self.chime_tmp))
            )
            self.chime.setVolume(0.45)

            self.rewind_timer = QTimer(self)
            self.rewind_timer.setInterval(80)
            self.rewind_timer.timeout.connect(
                self._rewind_step
            )

            self.was_playing_before_rewind = False
            self.rewinding = False

            self.slow_active = False
            self.fast_active = False

            self.btn_rewind = QPushButton("⏪")
            self.btn_rewind.setToolTip(
                "Mantener pulsado para rebobinar"
            )
            self.btn_rewind.pressed.connect(
                self._start_rewind
            )
            self.btn_rewind.released.connect(
                self._stop_rewind
            )

            self.btn_slow = QPushButton("0.5×")
            self.btn_slow.setCheckable(True)
            self.btn_slow.setToolTip(
                "Ralentizar reproducción"
            )
            self.btn_slow.toggled.connect(
                self._toggle_slow
            )

            self.btn_play = QPushButton("▶")
            self.btn_play.setToolTip("Play / pausa")
            self.btn_play.clicked.connect(
                self._toggle_play
            )

            self.btn_fast = QPushButton("2×")
            self.btn_fast.setCheckable(True)
            self.btn_fast.setToolTip(
                "Avance rápido"
            )
            self.btn_fast.toggled.connect(
                self._toggle_fast
            )

            controls = QHBoxLayout()
            controls.setContentsMargins(10, 8, 10, 8)
            controls.setSpacing(12)
            controls.addStretch(1)
            controls.addWidget(self.btn_rewind)
            controls.addWidget(self.btn_slow)
            controls.addWidget(self.btn_play)
            controls.addWidget(self.btn_fast)
            controls.addStretch(1)

            controls_widget = QWidget()
            controls_widget.setLayout(controls)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(controls_widget)
            layout.addWidget(self.video_overlay, 1)

            self.setCentralWidget(central)

            self.player.positionChanged.connect(
                self._position_changed
            )
            self.player.playbackStateChanged.connect(
                self._playback_state_changed
            )

            self.player.setSource(
                QUrl.fromLocalFile(
                    str(video_path.resolve())
                )
            )

            # Empieza reproduciendo.
            self.player.play()

        def modifier_active(self) -> bool:
            return (
                self.rewinding
                or self.slow_active
                or self.fast_active
            )

        def _event_count_at(self, position_ms: int) -> int:
            # Lista corta/mediana; recorrido incremental y simple.
            count = 0
            for event in self.events:
                if event.time_ms <= position_ms:
                    count += 1
                else:
                    break
            return count

        def _position_changed(self, position_ms: int) -> None:
            # No mostramos el trailer técnico de 8 frames del SHA-256.
            if (
                position_ms >= self.visual_end_ms
                and not self.rewinding
            ):
                self.visible_count = len(self.events)
                self.video_overlay.set_text(self.full_text)
                self.player.pause()
                if self.player.position() > self.visual_end_ms:
                    self.player.setPosition(self.visual_end_ms)
                return

            new_count = self._event_count_at(position_ms)

            # Si avanzamos normalmente, timbre por cada carácter nuevo.
            if (
                new_count > self.visible_count
                and not self.modifier_active()
                and self.player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ):
                for _ in range(new_count - self.visible_count):
                    self.chime.play()

            self.visible_count = new_count
            self.video_overlay.set_text(
                self.full_text[:new_count]
            )

        def _playback_state_changed(self, state) -> None:
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.btn_play.setText("⏸")
            else:
                self.btn_play.setText("▶")

        def _toggle_play(self) -> None:
            if self.rewinding:
                return

            if (
                self.player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ):
                self.player.pause()
            else:
                if self.player.position() >= self.visual_end_ms - 5:
                    self.player.setPosition(0)
                    self.visible_count = 0
                    self.video_overlay.set_text("")
                self.player.play()

        def _set_rate(self) -> None:
            if self.fast_active:
                self.player.setPlaybackRate(self.FAST_RATE)
            elif self.slow_active:
                self.player.setPlaybackRate(self.SLOW_RATE)
            else:
                self.player.setPlaybackRate(self.NORMAL_RATE)

        def _toggle_slow(self, checked: bool) -> None:
            self.slow_active = checked

            if checked and self.fast_active:
                self.btn_fast.blockSignals(True)
                self.btn_fast.setChecked(False)
                self.btn_fast.blockSignals(False)
                self.fast_active = False

            self._set_rate()

        def _toggle_fast(self, checked: bool) -> None:
            self.fast_active = checked

            if checked and self.slow_active:
                self.btn_slow.blockSignals(True)
                self.btn_slow.setChecked(False)
                self.btn_slow.blockSignals(False)
                self.slow_active = False

            self._set_rate()

        def _start_rewind(self) -> None:
            if self.rewinding:
                return

            self.rewinding = True

            self.was_playing_before_rewind = (
                self.player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            )

            self.player.pause()
            self.rewind_timer.start()

        def _stop_rewind(self) -> None:
            if not self.rewinding:
                return

            self.rewind_timer.stop()
            self.rewinding = False

            if self.was_playing_before_rewind:
                self.player.play()

        def _rewind_step(self) -> None:
            # ~4x hacia atrás: 320 ms cada 80 ms.
            new_position = max(
                0,
                self.player.position() - 320,
            )
            self.player.setPosition(new_position)

            if new_position == 0:
                self._stop_rewind()

        def closeEvent(self, event):
            self.player.stop()

            try:
                self.chime.stop()
            except Exception:
                pass

            try:
                self.chime_tmp.unlink(missing_ok=True)
            except Exception:
                pass

            super().closeEvent(event)

    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    restore_stderr = install_multimedia_stderr_filter()

    try:
        window = MainWindow()
        window.show()
        return app.exec() if owns_app else 0
    finally:
        restore_stderr()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvhide-decoder",
        description=(
            "Decodifica y verifica mensajes ocultos por UVHide Encoder 5x."
        ),
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Vídeo UVHide de entrada",
    )

    parser.add_argument(
        "--text",
        action="store_true",
        help=(
            "No abre el reproductor. Si el SHA-256 es válido, "
            "imprime únicamente el texto final."
        ),
    )

    parser.add_argument(
        "--delta0",
        type=int,
        default=DEFAULT_DELTA0,
        help="Diferencia usada por el encoder para bit 0 (por defecto: 3)",
    )

    parser.add_argument(
        "--delta1",
        type=int,
        default=DEFAULT_DELTA1,
        help="Diferencia usada por el encoder para bit 1 (por defecto: 7)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = decode_video(
        args.video,
        delta0=args.delta0,
        delta1=args.delta1,
    )

    if not result.hash_valid:
        die(
            "Se detectaron datos, pero el SHA-256 no coincide. "
            "El mensaje está corrupto o el vídeo no pertenece a este formato.",
            code=3,
        )

    if args.text:
        # Intencionadamente solo el texto en stdout.
        print(result.text)
        return

    launch_player(args.video, result)


if __name__ == "__main__":
    main()
