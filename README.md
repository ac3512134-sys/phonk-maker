# phonk-maker

Generate a short **phonk-style remix** from a WAV/MP3 input.

## Features
- Input validation (WAV/MP3, max 15 seconds)
- 0.75x slowdown + random pitch drop (2-4 semitones)
- BPM + beat-grid detection with `librosa`
- Strong bass EQ emphasis in the 20Hz-150Hz range
- Gritty distortion/saturation texture with ffmpeg
- Dark, wide atmospheric reverb
- Stereo widening tuned for headphone listening
- Louder punchy cowbell synced to detected BPM/first beat
- Clip-safe final level handling and WAV export
- Optional `--aggressive` mode for stronger effects

## Install
```bash
pip install librosa pydub soundfile numpy
# ffmpeg binary must be installed and on PATH
```

## Run
```bash
python phonk_remix.py input.wav --output output_phonk.wav
python phonk_remix.py input.wav --aggressive
```

If no output is provided, the module writes `output_phonk.wav`.
