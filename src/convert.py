from pydub import AudioSegment
import os


def convert_all_mp3_to_wav(directory: str = ".") -> None:
    """Convert all MP3 files in the given directory to WAV."""
    for filename in os.listdir(directory):
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(directory, filename)
            wav_path = os.path.join(directory, os.path.splitext(filename)[0] + ".wav")
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")
            print(f"Converted {mp3_path} -> {wav_path}")


if __name__ == "__main__":
    convert_all_mp3_to_wav()
