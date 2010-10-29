import numpy as np
import cuv_python as cp
import matplotlib
import math, os, pdb as pdb
from scipy.stats import scoreatpercentile

def showeights(x,i):
    import matplotlib.pyplot as plt
    plt.figure(i)
    #if (not ax):
        #plt.show()
    pltn=1
    trans=x.transpose()
    i=0
    maxp = 192
    ax=[]
    for step, row in enumerate(trans[200:200+maxp,:]):
        img = row[0:28*28].reshape(28,28)
        if (pltn-1 >= len(ax)):
            ax.append(plt.subplot(math.ceil(maxp / 16), maxp / math.ceil(maxp/16),  pltn))
        ax[pltn-1].set_axis_off()
        ax[pltn-1].matshow(img, cmap=plt.cm.bone_r)
        plt.title( str(step)  )
        #ax.imshow(img)
        pltn += 1
    plt.show()

def shomat(x):
    import matplotlib.pyplot as plt
    img = x[0:784,0].reshape(28,28)
    #img =x;
    plt.figure(1)
    ax = plt.gca()
    ax.matshow(img)
    ax.set_axis_off()
    plt.show()

def prmat(x,div=1.0):
    w = cp.pull(x)
    print w.min(), w.mean(), w.max()
    print "-----------------------------------------------"
    np.set_printoptions(threshold=sys.maxint, precision=2, suppress=True, linewidth=sys.maxint)
    print w/div

def lineno():
    return inspect.currentframe().f_back.f_back.f_lineno

def copy(x,y):
    cp.apply_binary_functor(x,y,cp.binary_functor.COPY)
def get_copy(x):
    y = x.__class__(x.h,x.w)
    copy(y.vec,x.vec)
    return y

def _matstr(x):
    "(%d,%d)[%2.1f Mb]"%(x.h,x.w,x.memsize/1024./1024.)
    
cp.dev_matrix_cmf.pull=cp.pull
cp.dev_matrix_cmf.__str__=_matstr
cp.dev_matrix_rmf.__str__=_matstr
cp.dev_dia_matrix_f.__str__=_matstr

def visualization_grid_size(num_imgs):
    v = math.sqrt(num_imgs)
    if v==int(v):
        return v,v
    else:
        return math.ceil(v),math.ceil(v)
    #while v > 3:
        #if float(num_imgs)/v == num_imgs/v and v < num_imgs/v:
            #return v, num_imgs/v
        #v-=1
    #raise ArithmeticError("can only draw square grids")

def flatten(seq):
    res = []
    for item in seq:
        if (isinstance(item, (tuple, list))):
            res.extend(flatten(item))
        else:
            res.append(item)
    return res

def cut_filters_func(px,py,maps,dividers=False):
    def func(x):
        x = x.reshape(py*maps,px)
        mean = x.mean()
        fact = x.max()
        L = []
        idx = [ px, 0, py, 0]
        for m in xrange(maps):
            x_ = x[m*py:(m+1)*py,:]
            sv = np.abs(x_).sum(axis=1)
            sh = np.abs(x_).sum(axis=0)
            sv = sv != 0
            if np.sum(sv) > 0:
                sh = sh != 0
                i2 = [ np.min(np.argwhere(sv)) , np.max(np.argwhere(sv)), np.min(np.argwhere(sh)) , np.max(np.argwhere(sh))]
                idx = [min(idx[0],i2[0]),
                       max(idx[1],i2[1]),
                       min(idx[2],i2[2]),
                       max(idx[3],i2[3])]
        for m in xrange(maps):
            x_ = x[m*py:(m+1)*py,:]
            sv = x_.sum(axis=1)
            sh = x_.sum(axis=0)
            x_ = x_[idx[0] : idx[1]+1 , idx[2] : idx[3]+1 ]
            L.append(x_)
        verdiv = fact * np.ones((idx[1]+1-idx[0],1))
        hordiv = fact * np.ones((1,idx[3]+1-idx[2]))
        for i in [4,2]:
            if len(L) % i == 0 and len(L)/i > 1:
                L2 = []
                while len(L):
                    # add vertical dividers
                    if dividers:
                        tmp = zip(L[0:i], [verdiv]*i)
                    else:
                        tmp = L[0:i]
                    L2.append( np.hstack( flatten(tmp) ) )
                    del L[0:i]
                L = L2
                hordiv = fact * np.ones((1,L[0].shape[1]))
                break
        if dividers:
            return np.vstack(flatten(zip(L, [hordiv]*len(L))))
        else:
            return np.vstack(L)
    return func

def visualize_rows(fig, mat, indices, row_to_img, title="", normalize=True, save=False, save_filename="grid.png", use_imshow=False, cb=False, separate_files=False):
    import matplotlib.pyplot as plt
    nrec = len(indices)
    ph, pw = visualization_grid_size(nrec)
    vmin = 1.0 * np.min(mat)
    vmax = 1.0 * np.max(mat)

    fig = plt.figure(fig)
    fig.canvas.set_window_title(title)
    plt.suptitle(title, fontsize=14) 
    norm = matplotlib.colors.Normalize()
    if normalize:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    if save and separate_files:
        for pltn, idx in enumerate(indices):
            img = row_to_img(mat[idx,:])
            if use_imshow:
                image=plt.imshow(img)
            else:
                image = plt.matshow(img, norm=norm, cmap=plt.cm.bone_r)
            if cb:
                plt.colorbar(image)
            plt.gca().set_axis_off()
            fig.subplots_adjust(bottom=0,top=1,left=0,right=1)
            ext = save_filename[-4:]
            fn = "%s_%04d"%(save_filename[0:-4] , pltn)
            plt.savefig(fn)
            plt.clf()
    else:
        for pltn, idx in enumerate(indices):
            ax = plt.subplot( ph, pw, 1+pltn)
            ax.set_axis_off()
            img = row_to_img(mat[idx,:])
            if use_imshow:
                image=ax.imshow(img)
            else:
                image = ax.matshow(img, norm=norm, cmap=plt.cm.bone_r)
            if cb:
                plt.colorbar(image)
        #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.125, right=0.9, wspace=0.05, hspace=0.05)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.25, right=0.85, wspace=0.05, hspace=0.05)
        plt.draw()
        if save: plt.savefig(save_filename,dpi=300)
        #if save: plt.savefig(save_filename)

def make_img_name(basename):
    usr=os.getenv("USER")
    try: os.mkdir(os.path.join("/tmp",usr))
    except OSError: pass
    return os.path.join("/tmp", usr, basename)

