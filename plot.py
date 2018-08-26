"""Plotting output quantities of iDEA
"""
from __future__ import division
from __future__ import print_function
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
ffmpeg_path = '/usr/bin/ffmpeg'


def read_quantity(pm, name):
    r"""Read a file from a pickle file in (/raw)

    parameters
    ----------
    pm:  parameters object
      parameters object
    name: string
      name of pickle file in (/raw) (e.g 'gs_ext_den')

    returns array_like
         data extracted from pickle file
    """
    input_file = open('raw/' + str(name) + '.db', 'rb')
    data = pickle.load(input_file)
    input_file.close()
    return np.array(data)


def to_data(pm, names, data, td, dim, file_name=None, timestep=0):
    r"""Outputs data to a .dat file in (/data)

    parameters
    ----------
    pm:  parameters object
      parameters object
    names: list of string
      names of the data to be saved (e.g 'gs_ext_den')
    data: list of array_like
      list of arrays to be plotted
    td: bool
      True: Time-dependent data, False: ground-state data
    dim: int
      number of dimentions of the data (0,1) (eg gs_ext_E would be 0, gs_ext_den would be 1)
    file_name: string
      name of output file (if None will be saved as default name e.g 'gs_ext_den.dat')
    timestep: int
      if td=True or data=3D specify the timestep to be saved
    """
    if len(data) > 1:
        raise IOError('cannot save multiple quantities to data file')
    data = data[0]
    names = names[0]
    file_name = names
    if td == False:
        if(dim == 0):
            d = [data]
            print('saving to data file...')
            np.savetxt('data/{}.dat'.format(file_name), d)
        if(dim == 1):
            x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
            d = np.zeros(shape=(len(x),2))
            d[:,0]=x[:]
            d[:,1]=data[:]
            print('saving to data file...')
            np.savetxt('data/{}.dat'.format(file_name), d)
    if td == True:
        if(dim == 0):
            d = [data[timestep]]
            print('saving to data file...')
            np.savetxt('data/{}.dat'.format(file_name), d)
        if(dim == 1):
            x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
            d = np.zeros(shape=(len(x),2))
            d[:,0]=x[:]
            d[:,1]=data[timestep,:]
            print('saving to data file...')
            np.savetxt('data/{0}_{1}.dat'.format(file_name, timestep), d)


def to_plot(pm, names, data, td, dim, file_name=None, timestep=0):
    r"""Outputs data to a .pdf file in (/plots)

    parameters
    ----------
    pm:  parameters object
      parameters object
    names: list of string
      names of the data to be saved (e.g 'gs_ext_den')
    data: list of array_like
      list of arrays to be plotted
    td: bool
      True: Time-dependent data, False: ground-state data
    dim: int
      number of dimentions of the data (0,1,2 or 3) (eg gs_ext_E would be 0, gs_ext_den would be 1)
    file_name: string
      name of output file (if None will be saved as default name e.g 'gs_ext_den.pdf')
    timestep: int
      if td=True or data=3D specify the timestep to be saved
    """
    x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
    if type(data) is not list:
        data = [data]
        names = [names]
    if file_name == None:
        file_name = '_'.join(names)
    if td == False:
        if(dim == 0):
            raise IOError('cannot plot 0D data to pdf')
        if(dim == 1):
            for n in names:
                print('adding {} to plot...'.format(n))
                plt.plot(x, data[names.index(n)], label=n)
            plt.xlabel('x (a.u.)')
            plt.ylabel('{} (a.u.)'.format(','.join(names)))
            plt.legend()
            print('saving plot to pdf...')
            plt.savefig('plots/{}.pdf'.format(file_name), dpi=500)
    if td == True:
        if(dim == 0):
            raise IOError('cannot plot time-dependent 0D data to pdf')
        if(dim == 1):
            for n in names:
                print('adding {} to plot...'.format(n))
                plt.plot(x, data[names.index(n)][timestep,:], label=n)
            plt.xlabel('x (a.u.)')
            plt.ylabel('{} (a.u.)'.format(','.join(names)))
            plt.legend()
            plt.title('timestep = {}'.format(timestep))
            print('saving plot to pdf...')
            plt.savefig('plots/{0}_{1}.pdf'.format(file_name, timestep), dpi=300)



def to_anim(pm, names, data, td, dim, file_name=None, step=1):
    r"""Outputs data to a .mp4 file in (/animations)

    parameters
    ----------
    pm:  parameters object
      parameters object
    names: list of string
      names of the data to be saved (e.g 'gs_ext_den')
    data: list of array_like
      list of arrays to be plotted
    td: bool
      True: Time-dependent data, False: ground-state data
    dim: int
      number of dimentions of the data (0,1,2 or 3) (eg gs_ext_E would be 0, gs_ext_den would be 1)
    file_name: string
      name of output file (if None will be saved as default name e.g 'gs_ext_den.dat')
    step: int
      number of frames to skip when animating
    """
    if type(data) is not list:
        data = [data]
        names = [names]
    if file_name == None:
        file_name = '_'.join(names)
    if td == False:
        if(dim == 0):
            raise IOError('cannot animate ground-state 0D quantities')
        if(dim == 1):
            raise IOError('cannot animate ground-state 1D quantities')
    if td == True:
        if(dim == 0):
            raise IOError('cannot animate time-dependent 0D quantity')
        if(dim == 1):
            from matplotlib import animation, rc
            ymax = np.max(data)+0.1*(np.max(data)-np.min(data))
            ymin = np.min(data)-0.1*(np.max(data)-np.min(data))
            fig, ax = plt.subplots()
            ax.set_xlim([-pm.sys.xmax,pm.sys.xmax])
            ax.set_ylim((ymin, ymax))
            array = []
            line = []
            print('animating... (may take some time)')
            for i in range(0, len(data)):
                array.append(data[i])
                lineObj, = ax.plot([], [], lw=2)
                line.append(lineObj)
            def init():
                for i in range(len(data)):
                    line[i].set_data([], [])
                return tuple(line)
            x = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
            def animate(i):
                for j in range(len(data)):
                    line[j].set_data(x, data[j][i,:])
                    line[j].set_label(names[j])
                    ax.set_title('timestep = {}'.format(i))
                legend = plt.legend()
                return tuple(line)
            mfile = "animations/{}.mp4".format(file_name)
            print("making movie {}... (may take some time)".format(mfile))
            rc('animation', html='html5')
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            frames = []
            for i in range(0, data[0].shape[0], int(step)):
                frames.append(i)
            frames = np.array(frames)
            line_ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=10, blit=True)
            writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            line_ani.save(mfile, writer=writer, dpi=300)
