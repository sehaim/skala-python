import os
import pickle
import subprocess

def bad():
    eval("print('hi')")
    os.system("ls -la")
    subprocess.run("whoami", shell=True)

def also_bad(data):
    return pickle.loads(data)
