"""Optional interactive recording; only records after explicit Enter per utterance."""
import argparse
from pathlib import Path
import soundfile as sf


def main():
    import sounddevice as sd  # optional: pip install sounddevice
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speaker', required=True)
    p.add_argument('--root', default='dataset')
    args = p.parse_args()
    if not args.speaker.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Speaker ID must contain only letters, digits, _ or -')
    words = {'mo':'mở','dong':'đóng','len':'lên','xuong':'xuống','dung':'dừng'}
    for label, word in words.items():
        for i in range(1,9):
            split = 'train' if i <= 6 else 'test'
            path = Path(args.root) / split / label / f'{args.speaker}_{label}_{i:02d}.wav'
            if path.exists():
                print('Already exists, skipping:', path)
                continue
            input(f'{word} ({i}/8): Enter to record 2 seconds; leave ~0.3 s silence at both ends. ')
            audio = sd.rec(32000, samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, 16000, subtype='PCM_16')
            print('Saved:', path)


if __name__ == '__main__':
    main()
