import subprocess
import sys
from ML.run_gen import generate_and_show

def make_dna(arg):
    if (arg[1] == ":"):
        return arg
    return [float(x) for x in arg.split(",")]

def make_image(args):
    generate_and_show(False, args[2], make_dna(args[-2]))
    return

def make_gif(args):
    generate_and_show(True, args[2], make_dna(args[-2]), make_dna(args[-1]), int(args[4]), int(args[3]))
    return

sub_process = subprocess.Popen(["npm", "run", "electron"], close_fds=True, shell=True, text=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

is_reading = False
s = ""

while sub_process.poll() is None:
    out = sub_process.stdout.read(1)
    if out == "?":
        if is_reading:
            args = s.split("|")
            if (args[1] == "IMG"):
                make_image(args)
            else:
                make_gif(args)
            s = ""
        is_reading = not is_reading
    if is_reading:
        s += out
    #sys.stdout.write(out)
    #sys.stdout.flush()

#result = subprocess.run(["npm", "run", "electron"], shell=True, stdout=subprocess.PIPE, text=True)
#print(result.stdout)