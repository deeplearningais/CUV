import matplotlib.pyplot as plt
import pylab
import numpy as np
import os, sys, re
from glob import glob

def get_data(file):
    L = []
    D = []
    for s in open(file,"r").readlines():
        it = re.split(r'\s+', s)
        print it
        D.append(it[0])
        L.append(float(it[1]))
    return np.array(L), D

def get_files(path):
    return glob(os.path.join(path, "*cuda*.dat"))

def plot(data, ax, off, width, color, label):
    xloc = np.arange(len(data))+off
    ax.bar(xloc,data,width=width,color=color,label=label)

def plot_all(reference, path):
    files = get_files(path)
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111)
    #ax.set_yscale('log')

    color_array = np.vstack(np.array(pylab.cm.datad["gist_rainbow"].values()))
    lc = len(color_array)
    lf = len(files)
    color_array = color_array[::lc/lf,:]
    color_array = ["r", "g", "b"]

    diff=0.7/len(files)
    off = 0.0
    color = 0
    refdata, refdesc = get_data(reference)
    for f in files:
        if os.path.samefile(f,reference):
            continue
        data, desc = get_data(f)
        data = refdata/data
        #plot(data,ax,off,diff,color_array[color,:],f)
        plot(data,ax,off,diff,color_array[color],f)
        off+=diff
        color+=1
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(desc)
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_rotation(70) 
    plt.title("Speed of GPU relative to "+reference)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_all(sys.argv[1],sys.argv[2])
