import os.path
import subprocess
import sys
import tempfile

import click
import cprint  # type: ignore
import numpy as np
import scipy.io.wavfile as wav
from numpy.typing import NDArray


# gong is a 2D array with shape [-1, 2]
def create_chunk(
    samples_per_sec: int, gong: NDArray[np.int16], duration_mins: int
) -> NDArray[np.int16]:
    duration_secs = duration_mins * 60
    gong_duration_secs = int(gong.shape[0] / samples_per_sec)
    silence_duration_secs = duration_secs - gong_duration_secs
    silence_n_samples = silence_duration_secs * samples_per_sec
    silence = np.zeros((silence_n_samples, 2), dtype=np.int16)
    data = np.vstack((gong, silence))
    return data


def check_input(
    gongfile: str | None, output: str | None, duration: int | None
) -> tuple[bool, str]:
    if gongfile is None or output is None or duration is None:
        return False, "All three parameters must be provided!"

    if not os.path.exists(gongfile):
        return False, f"{gongfile} does not exist!"

    if not gongfile.endswith(".mp3"):
        return False, f"{gongfile} must be an MP3 file!"

    if os.path.exists(output):
        return False, f"{output} already exists!"

    if not output.endswith(".mp3"):
        return False, f"{output} must be an MP3 file!"

    if duration <= 0:
        return False, f"duration must be a positive integer not {duration}!"

    return True, ""


@click.command()
@click.option("-g", "--gongfile", type=str, help="The gong MP3 file")
@click.option(
    "-o", "--output", type=str, help="The output file path (must end with MP3)"
)
@click.option(
    "-d",
    "--duration",
    type=int,
    help="The gong interval duration in minutes of the output",
)
def main(gongfile: str, output: str, duration: int):
    """
    Creates an MP3 file with silent padding between gongs. The start and end will be a gong.
    """

    is_valid, err_msg = check_input(gongfile, output, duration)
    if not is_valid:
        cprint.danger_print(err_msg)
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    cprint.info_print(f"Converting {gongfile} to WAV format.")
    tmpdir = tempfile.TemporaryDirectory()
    gong_wavfile = os.path.join(tmpdir.name, "gong.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", gongfile, gong_wavfile], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as err:
        cprint.danger_print("Unable to convert MP3 to WAV!")
        print(err.stderr)
        sys.exit(1)

    cprint.info_print("Generating silent wav.")
    rate, gong = wav.read(gong_wavfile)
    n_blocks = 60 // duration
    blocks = [create_chunk(rate, gong, duration) for _ in range(n_blocks)]
    blocks.append(gong)
    data = np.vstack(blocks)
    output_wavfile = os.path.join(tmpdir.name, "output.wav")
    wav.write(output_wavfile, rate, data)

    cprint.info_print("Converting silent to MP3 format.")
    try:
        subprocess.run(
            ["ffmpeg", "-i", output_wavfile, output], check=True, capture_output=True
        )
    except subprocess.CalledProcessError as err:
        cprint.danger_print("Unable to convert WAV to MP3!")
        print(err.stderr)
        sys.exit(1)
    cprint.success_print(f"Written {output}")


if __name__ == "__main__":
    main()
