import subprocess
import sys

sub_process = subprocess.Popen(["npm", "run", "electron"], close_fds=True, shell=True, text=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

is_reading = False
s = ""
while sub_process.poll() is None:
    out = sub_process.stdout.read(1)
    if out == ":":
        if is_reading:
            print(s)
            s = ""
        is_reading = not is_reading
    if is_reading:
        s += out
    #sys.stdout.write(out)
    #sys.stdout.flush()

#result = subprocess.run(["npm", "run", "electron"], shell=True, stdout=subprocess.PIPE, text=True)
#print(result.stdout)