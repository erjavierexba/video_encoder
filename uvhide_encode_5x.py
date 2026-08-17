#!/usr/bin/env python3
"""
UVHide Encoder
==============

Oculta un string dentro de un vídeo usando:

- 32 posiciones lógicas de datos distribuidas por la pantalla.
- Cada posición lógica usa 5 muestras físicas separadas en cruz.
- 1 posición lógica de control/cambio centrada exactamente en el centro
  absoluto del vídeo, también con 5 muestras.
- Cada estado de datos contiene 32 bits:
    * primer estado: longitud del mensaje
    * siguientes estados: un code point Unicode por carácter
    * 8 estados finales: SHA-256 completo (256 bits)
- El control alterna de estado cada vez que comienza una palabra/caracter nuevo.
- El vídeo se procesa frame a frame; no se carga entero en RAM.
- FFmpeg hace la decodificación/codificación; Python modifica únicamente
  33 posiciones lógicas = 165 muestras físicas por frame.

Uso:
    uvhide-encoder entrada.mp4 "HOLA"
    uvhide-encoder entrada.mp4 "HOLA" -o salida.mp4

Durante desarrollo:
    python3 uvhide_encode.py entrada.mp4 "HOLA"

Dependencias de desarrollo:
    ffmpeg / ffprobe
    numpy

La versión empaquetada puede llevar ffmpeg y ffprobe junto al ejecutable,
por lo que el usuario final no tiene por qué instalarlos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

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


# 32 posiciones lógicas normalizadas: 8 columnas × 4 filas.
# Se escalan automáticamente con la resolución.
DATA_COORDS = tuple(
    (x, y)
    for y in (0.12, 0.36, 0.64, 0.88)
    for x in (0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92)
)

HASH_WORDS = 8  # SHA-256 = 8 × 32 bits


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def bundled_base_dir() -> Path:
    """
    Directorio donde buscar binarios auxiliares.

    PyInstaller --onefile extrae los binarios empaquetados en sys._MEIPASS.
    En ejecución normal buscamos junto al script/ejecutable.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def find_binary(name: str) -> str:
    """
    Busca ffmpeg/ffprobe:
      1. Dentro del bundle o junto al ejecutable.
      2. En PATH.

    Esto permite distribuir un ejecutable autocontenido.
    """
    base = bundled_base_dir()

    candidates = [
        base / name,
        base / f"{name}.exe",
        base / "bin" / name,
        base / "bin" / f"{name}.exe",
    ]

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
        "en la versión distribuida debe ir incluido con UVHide Encoder."
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


def probe_video(path: Path) -> tuple[int, int, Fraction, int]:
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
        die("No he podido determinar los FPS del vídeo.")

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

    return width, height, fps, frame_count


def scaled_coordinates(width: int, height: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []

    for nx, ny in DATA_COORDS:
        x = int(round(nx * (width - 1)))
        y = int(round(ny * (height - 1)))
        coords.append((x, y))

    if len(set(coords)) != 32:
        die("La resolución es demasiado pequeña: algunas coordenadas se solapan.")

    return coords


def sample_spacing(width: int, height: int) -> int:
    """
    Separación de las cinco muestras de cada posición lógica.

    Aproximadamente:
      720p  -> 3 px
      1080p -> 4 px
      1440p -> 5 px
      2160p -> 8 px
    """
    return max(3, int(round(min(width, height) / 270.0)))


def sample_positions(
    x: int,
    y: int,
    spacing: int,
) -> tuple[tuple[int, int], ...]:
    """
    Cinco muestras físicas en cruz:

             X
         X   X   X
             X
    """
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
    all_logical = [*logical_coords, control_xy]
    all_physical: list[tuple[int, int]] = []

    for logical_x, logical_y in all_logical:
        for x, y in sample_positions(logical_x, logical_y, spacing):
            # Se necesita un margen de 1 px para leer los 8 vecinos.
            if not (1 <= x < width - 1 and 1 <= y < height - 1):
                die(
                    "La resolución es demasiado pequeña para distribuir "
                    "las muestras de forma segura."
                )
            all_physical.append((x, y))

    if len(set(all_physical)) != len(all_physical):
        die(
            "La resolución produce solapamiento entre muestras físicas. "
            "Prueba con una resolución mayor."
        )


def neighbor_mean_rgb(
    frame: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """
    Media RGB de los 8 píxeles inmediatamente vecinos,
    excluyendo el píxel central.
    """
    block = frame[y - 1:y + 2, x - 1:x + 2].astype(np.int16)
    total = block.sum(axis=(0, 1)) - block[1, 1]
    return total / 8.0


def encode_sample(
    frame: np.ndarray,
    x: int,
    y: int,
    bit: int,
    delta0: int,
    delta1: int,
) -> None:
    """
    Codifica una muestra física.

    El píxel se fuerza a una diferencia local:
      bit 0 -> delta0
      bit 1 -> delta1

    La dirección se elige independientemente por canal RGB para
    conservar margen incluso cerca de 0 o 255.
    """
    mean_rgb = neighbor_mean_rgb(frame, x, y)
    delta = delta1 if bit else delta0

    directions = np.where(mean_rgb < 128.0, 1.0, -1.0)
    target = np.rint(mean_rgb + directions * delta)

    frame[y, x] = np.clip(target, 0, 255).astype(np.uint8)


def encode_bit(
    frame: np.ndarray,
    x: int,
    y: int,
    bit: int,
    spacing: int,
    delta0: int,
    delta1: int,
) -> None:
    """
    Codifica un bit lógico en cinco muestras físicas separadas.

    El decoder decidirá por mayoría:
      5/5
      4/5
      3/5
    """
    for sample_x, sample_y in sample_positions(x, y, spacing):
        encode_sample(
            frame,
            sample_x,
            sample_y,
            bit,
            delta0,
            delta1,
        )


def word_to_bits(word: int) -> list[int]:
    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError(f"Palabra fuera de rango uint32: {word}")

    return [
        (word >> shift) & 1
        for shift in range(31, -1, -1)
    ]


def encode_word(
    frame: np.ndarray,
    word: int,
    control_state: int,
    coords: list[tuple[int, int]],
    control_xy: tuple[int, int],
    spacing: int,
    delta0: int,
    delta1: int,
) -> None:
    bits = word_to_bits(word)

    for (x, y), bit in zip(coords, bits):
        encode_bit(
            frame,
            x,
            y,
            bit,
            spacing,
            delta0,
            delta1,
        )

    control_x, control_y = control_xy
    encode_bit(
        frame,
        control_x,
        control_y,
        control_state,
        spacing,
        delta0,
        delta1,
    )


def sha256_words(message: str) -> list[int]:
    """
    SHA-256 sobre:
        uint32_be(longitud_en_caracteres) || mensaje_UTF8
    """
    payload = (
        len(message).to_bytes(4, "big")
        + message.encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()

    return [
        int.from_bytes(digest[i:i + 4], "big")
        for i in range(0, 32, 4)
    ]


def word_for_source_frame(
    frame_index: int,
    source_frames: int,
    message: str,
) -> tuple[int, int]:
    """
    Frame 0:
      longitud del mensaje, control=0.

    Frames 1..N-1:
      caracteres repartidos entre todos los frames restantes.

    La duración de un carácter puede ser de un único frame o de muchos.
    El control alterna cada vez que empieza un carácter nuevo.
    """
    if frame_index == 0:
        return len(message), 0

    available = source_frames - 1
    char_count = len(message)

    char_index = min(
        ((frame_index - 1) * char_count) // available,
        char_count - 1,
    )

    word = ord(message[char_index])
    control_state = (char_index + 1) & 1

    return word, control_state


def fps_arg(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def encode_video(
    input_path: Path,
    output_path: Path,
    message: str,
    delta0: int,
    delta1: int,
    preset: str,
) -> None:
    global FFMPEG, FFPROBE

    init_external_tools()
    assert FFMPEG is not None
    assert FFPROBE is not None

    if not input_path.exists():
        die(f"No existe: {input_path}")

    if not message:
        die("El mensaje no puede estar vacío.")

    if len(message) > 0xFFFFFFFF:
        die("El mensaje es demasiado largo.")

    for char in message:
        cp = ord(char)
        if 0xD800 <= cp <= 0xDFFF:
            die("El mensaje contiene un surrogate Unicode no válido.")

    if not (0 <= delta0 < delta1 <= 64):
        die("Debe cumplirse 0 <= delta0 < delta1 <= 64.")

    width, height, fps, source_frames = probe_video(input_path)

    if source_frames < 2:
        die("El vídeo necesita al menos 2 frames.")

    if source_frames - 1 < len(message):
        die(
            "No hay suficientes frames para el mensaje.\n"
            f"Máximo: {source_frames - 1} caracteres\n"
            f"Solicitados: {len(message)}"
        )

    coords = scaled_coordinates(width, height)

    # Centro lógico exacto de la rejilla de píxeles.
    control_xy = (width // 2, height // 2)

    spacing = sample_spacing(width, height)

    validate_sample_positions(
        width,
        height,
        coords,
        control_xy,
        spacing,
    )

    hash_words = sha256_words(message)
    frame_bytes = width * height * 3

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Entrada:        {input_path}")
    print(f"Salida:         {output_path}")
    print(f"Resolución:     {width}x{height}")
    print(f"FPS:            {float(fps):.6g}")
    print(f"Frames fuente:  {source_frames}")
    print(f"Caracteres:     {len(message)}")
    print(f"Datos:          32 bits × 5 muestras")
    print(f"Control:        5 muestras en centro {control_xy}")
    print(f"Separación:     {spacing}px")
    print(f"Muestras/frame: 165")
    print(f"Hash:           SHA-256 ({HASH_WORDS} frames finales)")
    print()

    decoder = subprocess.Popen(
        [
            FFMPEG,
            "-v", "error",
            "-i", str(input_path),
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

    with tempfile.TemporaryDirectory(prefix="uvhide_") as tmp_dir:
        tmp_video = Path(tmp_dir) / "encoded_video.mkv"

        encoder = subprocess.Popen(
            [
                FFMPEG,
                "-y",
                "-v", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-video_size", f"{width}x{height}",
                "-framerate", fps_arg(fps),
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264rgb",
                "-crf", "0",
                "-preset", preset,
                str(tmp_video),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if encoder.stdin is None:
            decoder.kill()
            die("No se pudo abrir la entrada del encoder FFmpeg.")

        last_clean_frame: np.ndarray | None = None
        processed = 0
        next_progress = 10

        try:
            for frame_index in range(source_frames):
                raw = decoder.stdout.read(frame_bytes)

                if len(raw) != frame_bytes:
                    break

                frame = np.frombuffer(
                    raw,
                    dtype=np.uint8,
                ).reshape(
                    (height, width, 3)
                ).copy()

                # Guardamos el último frame SIN modificar para usarlo
                # como imagen base de los 8 frames del hash.
                if frame_index == source_frames - 1:
                    last_clean_frame = frame.copy()

                word, control_state = word_for_source_frame(
                    frame_index,
                    source_frames,
                    message,
                )

                encode_word(
                    frame,
                    word,
                    control_state,
                    coords,
                    control_xy,
                    spacing,
                    delta0,
                    delta1,
                )

                encoder.stdin.write(frame.tobytes())
                processed += 1

                percent = int(processed * 100 / source_frames)
                if percent >= next_progress:
                    print(f"Procesando: {min(percent, 100)}%")
                    next_progress += 10

            if processed != source_frames:
                die(
                    "FFmpeg devolvió un número inesperado de frames: "
                    f"{processed}/{source_frames}"
                )

            if last_clean_frame is None:
                die("No se pudo obtener el último frame.")

            # Trailer SHA-256: 8 palabras de 32 bits = 8 frames.
            for hash_index, word in enumerate(hash_words):
                trailer_frame = last_clean_frame.copy()

                control_state = (
                    len(message)
                    + hash_index
                    + 1
                ) & 1

                encode_word(
                    trailer_frame,
                    word,
                    control_state,
                    coords,
                    control_xy,
                    spacing,
                    delta0,
                    delta1,
                )

                encoder.stdin.write(trailer_frame.tobytes())

            encoder.stdin.close()

        except BrokenPipeError:
            pass

        decoder_rc = decoder.wait()
        encoder_rc = encoder.wait()

        decoder_err = (
            decoder.stderr.read().decode("utf-8", "replace")
            if decoder.stderr
            else ""
        )

        encoder_err = (
            encoder.stderr.read().decode("utf-8", "replace")
            if encoder.stderr
            else ""
        )

        if decoder_rc != 0:
            die(f"FFmpeg falló decodificando:\n{decoder_err.strip()}")

        if encoder_rc != 0:
            die(f"FFmpeg falló codificando:\n{encoder_err.strip()}")

        # Vídeo lossless nuevo + audio del original.
        remux_cmd = [
            FFMPEG,
            "-y",
            "-v", "error",
            "-i", str(tmp_video),
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-map_metadata", "1",
            "-c:v", "copy",
        ]

        suffix = output_path.suffix.lower()

        if suffix in {".mkv", ".mka"}:
            remux_cmd += ["-c:a", "copy"]

        elif suffix in {".mp4", ".m4v", ".mov"}:
            # AAC garantiza una salida habitual en MP4/MOV aunque
            # el audio de entrada use otro codec.
            remux_cmd += [
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
            ]

        else:
            remux_cmd += ["-c:a", "copy"]

        remux_cmd.append(str(output_path))

        remux = subprocess.run(
            remux_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if remux.returncode != 0:
            die(
                "El vídeo se codificó, pero falló el remux final:\n"
                + remux.stderr.strip()
            )

    digest_hex = hashlib.sha256(
        len(message).to_bytes(4, "big")
        + message.encode("utf-8")
    ).hexdigest()

    print()
    print("OK: vídeo codificado.")
    print(f"SHA-256: {digest_hex}")
    print(f"Frames de salida: {source_frames + HASH_WORDS}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvhide-encoder",
        description=(
            "Oculta texto Unicode en un vídeo mediante 32 posiciones "
            "lógicas redundantes y un control central."
        ),
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Vídeo de entrada",
    )

    parser.add_argument(
        "mensaje",
        help="Texto que se quiere ocultar",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Vídeo de salida. Por defecto conserva la extensión: "
            "<entrada>_uvhidden.<ext>"
        ),
    )

    parser.add_argument(
        "--delta0",
        type=int,
        default=3,
        help="Diferencia local para bit 0 (por defecto: 3)",
    )

    parser.add_argument(
        "--delta1",
        type=int,
        default=7,
        help="Diferencia local para bit 1 (por defecto: 7)",
    )

    parser.add_argument(
        "--preset",
        default="fast",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="Preset de libx264rgb (por defecto: fast)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.video

    if args.output:
        output_path: Path = args.output
    else:
        suffix = input_path.suffix or ".mp4"
        output_path = input_path.with_name(
            input_path.stem
            + "_uvhidden"
            + suffix
        )

    encode_video(
        input_path=input_path,
        output_path=output_path,
        message=args.mensaje,
        delta0=args.delta0,
        delta1=args.delta1,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
