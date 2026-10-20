"""Create explicitly SYNTHETIC Vietnamese command audio; no microphone use."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WORDS = {'mo': 'mở', 'dong': 'đóng', 'len': 'lên', 'xuong': 'xuống', 'dung': 'dừng'}


def generate(root):
    import espeakng_loader
    root = Path(root)
    if list(root.rglob('*.wav')):
        raise FileExistsError(f'{root} already contains WAV files; refusing to overwrite.')
    lib = espeakng_loader.load_library()
    if lib is None:
        raise RuntimeError('Cannot load eSpeak NG')
    lib.espeak_Initialize.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.espeak_Initialize.restype = C.c_int
    rate = lib.espeak_Initialize(2, 0, espeakng_loader.get_data_path().encode(), 0)
    if rate <= 0:
        raise RuntimeError('eSpeak initialization failed')
    callback_type = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.c_void_p)
    pieces = []

    @callback_type
    def callback(wav, count, events):
        if wav and count > 0:
            pieces.append(np.ctypeslib.as_array(wav, shape=(count,)).copy())
        return 0

    lib.espeak_SetSynthCallback.argtypes = [callback_type]
    lib.espeak_SetSynthCallback(callback)
    lib.espeak_SetVoiceByName.argtypes = [C.c_char_p]
    if lib.espeak_SetVoiceByName(b'vi') != 0:
        raise RuntimeError('Vietnamese voice vi unavailable')
    lib.espeak_Synth.argtypes = [C.c_void_p, C.c_size_t, C.c_uint, C.c_int,
                                C.c_uint, C.c_uint, C.c_void_p, C.c_void_p]
    settings = {'train': [(135,38),(150,44),(165,50),(180,56),(195,62),(210,68)],
                'test': [(158,47),(188,59)]}
    records = []
    for split, variants in settings.items():
        for label, word in WORDS.items():
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for index, (speed, pitch) in enumerate(variants, 1):
                pieces.clear()
                lib.espeak_SetParameter(1, speed, 0)
                lib.espeak_SetParameter(3, pitch, 0)
                raw = word.encode('utf-8') + b'\0'
                buffer = C.create_string_buffer(raw)
                status = lib.espeak_Synth(buffer, len(raw), 0, 1, 0, 1, None, None)
                if status or not pieces:
                    raise RuntimeError(f'Synthesis failed: {word}')
                y = np.concatenate(pieces).astype(np.float64) / 32768
                y = resample_poly(y, 16000, rate)
                y = np.pad(y, (4000, 4000))
                path = folder / f'{label}_{index:02d}.wav'
                sf.write(path, y, 16000, subtype='PCM_16')
                records.append(dict(path=path.relative_to(root).as_posix(), label=label,
                                    text=word, speed=speed, pitch=pitch,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    lib.espeak_Terminate()
    metadata = dict(kind='synthetic_espeak_ng', voice='vi', human_recordings=False,
                    description='40 separate syntheses; shared engine/voice across splits. '
                                'NOT independent human speakers. Accuracy only demonstrates software.',
                    source='https://github.com/espeak-ng/espeak-ng', records=records)
    (root / 'PROVENANCE.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Created {len(records)} SYNTHETIC utterances in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dataset_demo')
    generate(parser.parse_args().output)
