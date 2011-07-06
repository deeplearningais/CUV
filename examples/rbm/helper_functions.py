import numpy as np
import matplotlib
import math, os

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
            fn = "%s_%04d"%(save_filename[0:-4] , pltn)
            plt.savefig(fn)
            plt.clf()
    else:
        for pltn, idx in enumerate(indices):
            ax = plt.subplot( int(ph), int(pw), 1+pltn)
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

