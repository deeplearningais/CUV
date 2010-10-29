import cPickle
import sys, os


found = 0
arg  = sys.argv[1]
base = sys.argv[2]
file = sys.argv[3]
fn = os.path.join(base,file)
if os.path.exists(fn):
    with open(fn, "r") as f:
        x = cPickle.load(f)
        print x.keys()
    if arg in x:
        print fn, "\t", x[arg]
        sys.exit(1)
else:
    print "File does not exist!"
sys.exit(0)
