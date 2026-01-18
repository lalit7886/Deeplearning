import os
import regex as re
import subprocess
import urllib
import numpy as np
from IPython.display import Audio


cwd=os.path.dirname(__file__)

def load_training_data():
    with open(os.path.join(cwd,"irish.abc"),"r") as f:
        text= f.read()
        songs=extract_songs_snippet(text)
    return songs

def extract_songs_snippet(text):
    pattern="(^|\n\n)(.*?)\n\n"
    search_results=re.findall(pattern=pattern,string=text,overlapped=True,flags=re.DOTALL)
    songs=[songs[1] for songs in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs
    
def save_song_to_abc(song,file_name="temp"):
    save_name="{}.abc".format(file_name)
    save_path=os.path.join(cwd,save_name)
    with open(save_path,"w") as f:
        f.write(song)
    print(f"file is saved to path {save_path}")
    return file_name

def abc2wav(abc_file):
    path_to_tool=os.path.join(cwd,"bin","abc2wav")
    abc_file="{}.abc".format(abc_file)
    data_path=os.path.join(cwd,abc_file)
    if not os.path.isfile(path_to_tool):
        raise FileNotFoundError("the file is not found error occured")
    cmd="{} {}".format(path_to_tool,data_path)
    return os.system(cmd)

def play_wave(wave_file):
    return Audio(filename=wave_file,autoplay=True)

def play_song(song):
    basename=save_song_to_abc(song)
    ret=abc2wav(basename)
    mp_path=os.path.join(cwd,basename)
    if ret==0:
        return play_wave(mp_path+".wav")
    return None


def play_generated_song(generated_text):
    songs=extract_songs_snippet(generated_text)
    if len(songs)==0:
        print("No valid song is found in generated text")
    for song in songs:
        play_song(song)

    print(
        "None of the songs were valid, try training longer to improve \
        syntax."
    )

def test_batch_func_type(func,args):
    ret=func(*args)
    assert len(ret)==2, "[FAIL] get batch must return two argumnets input and labels"
    
    assert type(ret[0])==np.ndarray, "[FAIL] test_batch_func_type x: is of type np.array"
    assert type(ret[1])==np.ndarray, "[FAIL] test_batch_func_type x: is of type np.array"
    print("test_batch_fucntion is passed")
    
    return True

def test_batch_func_next_step(func,args):
    x,y=func(*args)
    assert (
        x[:, 1:] == y[:, :-1]
    ).all(), "[FAIL] test_batch_func_next_step: x_{t} must equal y_{t-1} for all t"
    print("[PASS] test_batch_func_next_step")
    return True

    
    