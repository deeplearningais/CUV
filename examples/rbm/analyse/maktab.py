#!/usr/bin/python 
import sys
import matplotlib.pyplot as plt
import cPickle
import os
import numpy as np
from glob import glob

def safeacc(f,s):
    if f in s:
        return s[f]
    else:
        return np.nan
def main():
    cols = ["basename", "bindex", "flipenergyloss", "ais_nats","ais_z_std", "reconstruction", "nats", "cpf_host", "cpf_duration", "dev", "ais_host", "ais_dev", "ais_duration", "weights2","reconstruction"]
    numerics = ["ais_nats", "reconstruction", "nats","weights2","ais_z_std"]
    show = ["nats", "ais_nats"]
    if len(sys.argv) < 2:
        print "need base dir"
        sys.exit(1)
    base = sys.argv[1]
    print "# ", "\t".join([x for x in cols])
    lists=dict()
    for x in cols:
        lists[x]=dict()

    for fn in sorted(glob(os.path.join(base,'info-0-*.pickle'))):
        with open(fn, "r") as f:
            stats = cPickle.load(f)
        try:
            x = int(safeacc("bindex", stats))
        except:
            continue
        if x < 0:
            continue
        if   "ais_p_nats_14500" in stats:
            stats["ais_nats"] = stats["ais_p_nats_14500"]
        elif "ais_p_nats_10000" in stats:
            stats["ais_nats"] = stats["ais_p_nats_10000"]
        elif "ais_p_nats_1000" in stats:
            stats["ais_nats"] = stats["ais_p_nats_1000"]
        
        if   "ais_z_std_14500" in stats:
            stats["ais_z_std"] = stats["ais_z_std_14500"]
        elif "ais_z_std_10000" in stats:
            stats["ais_z_std"] = stats["ais_z_std_10000"]
        elif "ais_z_std_1000" in stats:
            stats["ais_z_std"] = stats["ais_z_std_1000"]

        if   "weights2" in stats:
            stats["weights2"] = np.sqrt(stats["weights2"])
        print "\t".join([str(safeacc(x,stats)) for x in cols])

        v = map (lambda x:[x,safeacc(x,stats)],numerics) 
        # save the keys as well as maped values
        v=dict(v)
        # convert list of tuples to dict

        i  = int(safeacc("bindex", stats))

        # key denotes the quiantity, i is the current iteration.
        # each lists[key] is a list of all values for the different seeds
        for key,value in v.iteritems():
            if value==value:
                # value is not nan
                if i in lists[key]:
                    lists[key][i].append(value)
                else:
                    lists[key][i]=[value]


    indices=dict()
    means=dict()
    stds=dict()


    for key in numerics:
        # indices[keys] stores iterations where we have values for quantity key
        indices[key]=lists[key].keys()
        indices[key].sort()
        means[key]=[np.array(lists[key][x]).mean() for x in indices[key]] # mean over trials
        stds[key]=[np.array(lists[key][x]).std() for x in indices[key]] # std over trials

        
    ax = plt.subplot(111)

    for key in show:
        if len(indices[key]):
            ax.errorbar(indices[key],means[key],stds[key], label=key)

    #write first column to file
    #with open("table-"+base[:-1]+".txt","w") as f:
        ## column headers
        #f.write("index ")
        #for key in numerics:
            #f.write("%s_mean %s_std "%(key,key))
        #f.write("\n")
        ## values
        #for index,iteration in enumerate(indices[numerics[0]]):
            #f.write("%d "%iteration)
            #for key in numerics:
                #f.write("%f %f "%(means[key][index],stds[key][index]))
            #f.write("\n")

     
    #plt.ylim(-300,-120)
    #plt.legend(loc='top left')
    #plt.legend(loc="upper left")
    #ax2 = plt.twinx()
    #ax2 = ax
    #ax2.plot(I2,L2, "g-", label=show[2])
    plt.title(base)
    plt.legend(loc="lower right")
    #plt.savefig("figure-"+base+".png")
    with open("statistics-"+base+".pickle","wc") as f:
        cPickle.dump(dict(indices=indices,means=means,stds=stds,lists=lists),f)
    plt.show()

if __name__ == "__main__":
    main()

