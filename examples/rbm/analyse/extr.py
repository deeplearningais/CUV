import cPickle
import sys, os


for i in sys.argv[2:]:
    fn = os.path.join(i,"info-0.pickle")
    arg = sys.argv[1]
    if os.path.exists(fn):
        with open(fn, "r") as f:
            x = cPickle.load(f)
        if arg in x:
            print i, "\t", x[arg]
    fn = os.path.join(i,"cfg.pickle")
    if os.path.exists(fn):
        with open(fn, "r") as f:
            x = cPickle.load(f)
        if arg in x:
            print i, "\t", x[arg]
