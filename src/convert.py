from os import path
from pydub import AudioSegment

num = 7 
for i in range(1, 5):
    num +=  1

    src = f"{num}.mp3"
    dst = f"{num}.wav"

    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")