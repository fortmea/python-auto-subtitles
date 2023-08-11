import sys
import librosa
from librosa import display
import numpy as np
import IPython.display as ipd
import matplotlib as plt
import shlex
import subprocess
import os
import soundfile as sf
from audio_separator import Separator
global original_file
original_file = sys.argv[1]
def main():
    original_file = sys.argv[1]
    if os.path.exists(original_file):
        print("OK")
    original_file = os.path.abspath(original_file)
    print(original_file)
    splitvideo()
    splitaudio()
    createsub()
    addsub()
    #file_to_write = sys.argv[2] 
    #use_cuda = sys.argv[4] 
    #font = sys.argv[3] if len(sys.argv) > 3 else None
    
def addsub():
    #add subtitles
    command_sub = 'ffmpeg -y -i .\output-audio_(Vocals)_UVR-MDX-NET-Inst_HQ_1.srt sub.ass'
    print(command_sub)
    subprocess.run(command_sub)
    command_sub = 'ffmpeg -y -i "'+ original_file + '" -vf '+ "subtitles=sub.ass:force_style='Fontname=Consolas,BackColour=&H80000000,Spacing=0.2,Outline=0,Shadow=0.75' " +'"' + original_file + '_sub.mp4"'
    print(command_sub)
    subprocess.run(command_sub)
    
def splitvideo():
    command = 'ffmpeg.exe -y -i "' + original_file + '" -vn -acodec copy output-audio.aac'
    print(command)
    subprocess.run(command)

def splitaudio():
     # Initialize the Separator with the audio file and model name
    separator = Separator('output-audio.aac', model_name="UVR-MDX-NET-Inst_HQ_1",  use_cuda=True)
    # Perform the separation
    global primary_stem_path
    global secondary_stem_path 
    primary_stem_path, secondary_stem_path = separator.separate()
    
def createsub():
     # create transcription
    command_whisperx = "whisperx " + secondary_stem_path + " --model large-v2 --batch_size 4 --max_line_width 36 --highlight_words True --max_line_count 1"
    print(command_whisperx)
    subprocess.run(command_whisperx)
if __name__ == "__main__":
    main()