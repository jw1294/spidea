""" Stores input parameters for iDEA calculations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from six import string_types
import results as rs
import numpy as np
import importlib
import os
import sys
import copy
import time
import SPiDEA


def input_string(key,value):
    """Prints a line of the input file"""
    if isinstance(value, string_types):
        s = "{} = '{}'\n".format(key, value)
    else:
        s = "{} = {}\n".format(key, value)
    return s


class InputSection(object):
   """Generic section of input file"""

   def __str__(self):
       """Print variables of section and their values"""
       s = ""
       v = vars(self)
       for key,value in v.items():
           s += input_string(key, value)
       return s


class SystemSection(InputSection):
    """System section of input file

    Includes some derived quantities.
    """

    @property
    def deltax(self):
        """Spacing of real space grid"""
        return 2.0*self.xmax/(self.grid-1)

    @property
    def deltat(self):
        """Spacing of temporal grid"""
        return 1.0*self.tmax/(self.imax-1)

    @property
    def ideltat(self):
        """Spacing of temporal grid"""
        return 1.0*self.itmax/(self.iimax-1)


class Input(object):
    """Stores variables of input parameters file

    Includes automatic generation of dependent variables,
    checking of input parameters, printing back to file and more.
    """

    priority_dict = {
      'low': 2,
      'default': 1,
      'high': 0}

    def __init__(self):
        """Sets default values of some properties."""
        self.filename = ''
        self.log = ''
        self.last_print = time.clock()

        ### Run parameters
        run = InputSection()
        run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\)
        run.time_dependence = False          #: Run time-dependent calculation
        run.verbosity = 'default'            #: Output verbosity ('low', 'default', 'high')
        run.save = True                      #: Save results to disk when they are generated
        self.run = run


        ### System parameters
        sys = SystemSection()
        sys.grid = 1001                      #: Number of grid points (must be odd)
        sys.xmax = 10.0                      #: Size of the system
        sys.tmax = 1.0                       #: Total real time
        sys.imax = 1001                      #: Number of real time iterations (NB: deltat = tmax/(imax-1))
        sys.acon = 1.0                       #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1.0       #: Scales the strength of the Coulomb interaction

        def v_ext(x):
            """Ground-state external potential
            """
            return 0.5*(0.25**2)*(x**2)
        sys.v_ext = v_ext

        def v_pert(x):
            """Perturbing potential (switched on at t=0)
            """
            return -0.01*x
        sys.v_pert = v_pert
        self.sys = sys


        ### SPiDEA parameters
        ext = InputSection()
        ext.itol = 1e-12                     #: Tolerance of imaginary time propagation (Recommended: 1e-12)
        ext.itol_solver = 1e-14              #: Tolerance of linear solver in imaginary time propagation (Recommended: 1e-14)
        ext.rtol_solver = 1e-12              #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
        ext.itmax = 2000.0                   #: Total imaginary time
        ext.iimax = 1e5                      #: Imaginary time iterations
        ext.ideltat = ext.itmax/ext.iimax    #: Imaginary time step (DERIVED)
        self.ext = ext


    def check(self):
        """Checks validity of input parameters."""
        pm = self
        if pm.ext.itol > 1e-6:
            self.sprint('EXT: Warning - value of ext.itol is much larger than 1e-13, this can yeild poor densities')


    def __str__(self):
        """Prints different sections in input file"""
        s = ""
        v = vars(self)
        for key, value in v.items():
            if isinstance(value, InputSection):
                s += "### {} section\n".format(key)
                s += "{}\n".format(value)
            else:
                s += input_string(key,value)
        return s


    def sprint(self, string='', priority=1, newline=True, refresh=0.000005, savelog=True):
        """Customized print function

        Prints to screen and appends to log.

        If newline == False, overwrites last line,
        but refreshes only every refresh seconds.

        parameters
        ----------
        string : string
            string to be printed
        priority: int
            priority of message, possible values are
            0: debug
            1: normal
            2: important
        newline : bool
            If False, overwrite the last line
        refresh : float
            If newline == False, print only every "refresh" seconds
        savelog : bool
            If True, save string to log file
        """
        verbosity = self.run.verbosity
        if(savelog):
            self.log += string + '\n'
        if priority >= self.priority_dict[verbosity]:

            timestamp = time.clock()
            if newline:
                print(string)
                self.last_print = timestamp
            # When overwriting lines, we only print every "refresh" seconds
            elif timestamp - self.last_print > refresh:
                ## this only overwrites, no erase
                #print('\r' + string, end='')

                # Overwrite line
                sys.stdout.write('\r' + string)
                # Delete rest of line starting from cursor position (in case
                # previous line was longer). See
                # https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
                sys.stdout.write(chr(27) + '[K')
                sys.stdout.flush()

                self.last_print = timestamp
            else:
                pass


    @classmethod
    def from_python_file(cls,filename):
        """Create Input from Python script."""
        tmp = Input()
        tmp.read_from_python_file(filename)
        return tmp


    def read_from_python_file(self, filename):
        """Update Input from Python script."""
        if not os.path.isfile(filename):
            raise IOError("Could not find file {}".format(filename))

        module, ext = os.path.splitext(filename)
        if ext != ".py":
            raise IOError("File {} does not have .py extension.".format(filename))

        # import module into object
        pm = importlib.import_module(module)

        # Replace default member variables with those from parameters file.
        # The following recursive approach is adapted from
        # See http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def update(d, u, l=1):
            for k, v in u.items():
                # We need to step into InputSection objects, as those may have varying
                # numbers of parameters defined.
                if isinstance(v, InputSection):
                    r = update(d.get(k, {}).__dict__, v.__dict__, l+1)
                    d[k].__dict__ = r
                    #d[k] = r
                # We only want to copy contents of the input sections
                # No need to copy any of the builtin attributes added
                elif l > 1:
                    d[k] = u[k]
            return d

        self.__dict__ = update(self.__dict__, pm.__dict__)

        self.filename = filename


    @property
    def output_dir(self):
        """Returns full path to output directory
        """
        return 'outputs/{}'.format(self.run.name)


    def make_dirs(self):
        """Set up ouput directory structure"""
        import os
        import shutil
        import errno
        pm = self

        def mkdir_p(path):
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else: raise
        output_dirs = ['data', 'raw', 'plots', 'animations']
        for d in output_dirs:
            path = '{}/{}'.format(pm.output_dir,d)
            mkdir_p(path)
            setattr(pm,d,path)
        # Copy parameters file to output folder, if there is one
        if os.path.isfile(pm.filename):
            shutil.copy2(pm.filename,pm.output_dir)
        # Copy ViDEO file to output folder
        vfile = 'ViDEO.py'
        if os.path.isfile(vfile):
            # Note: this doesn't work, when using iDEA as a system module
            shutil.copy2('ViDEO.py',pm.output_dir)
        vfile = 'plot.py'
        if os.path.isfile(vfile):
            # Note: this doesn't work, when using iDEA as a system module
            shutil.copy2('plot.py',pm.output_dir)
        else:
            pass
            # No longer needed as ViDEO.py is in scrips directory and can be added to PATH
            #s  = "Warning: Unable to copy ViDEO.py since running iDEA as python module."
            #s += " Simply add the scripts folder to your PATH variable to use ViDEO.py anywhere"
            #pm.sprint(s,1)


    def execute(self):
        """Run this job"""
        pm = self
        pm.check()
        if pm.run.save:
            pm.make_dirs()
        self.results = rs.Results()

        # Draw splash to screen
        print('                                                                   ')
        print('          ****  ****   *    ****     *****        *                ')
        print('         *      *   *       *   *    *           * *               ')
        print('         *      *   *  *    *    *   *          *   *              ')
        print('          ***   ****   *    *     *  *****     *******             ')
        print('             *  *      *    *    *   *        *       *            ')
        print('             *  *      *    *   *    *       *         *           ')
        print('         ****   *      *    ****     *****  *           *          ')
        print('                                                                   ')
        print('  +------------------------------------------------------------+   ')
        print('  |                    Single-Particle iDEA                    |   ')
        print('  |                                                            |   ')
        print('  |                   Created by Jack Wetherell                |   ')
        print('  |                    The University of York                  |   ')
        print('  +------------------------------------------------------------+   ')
        print('                                                                   ')
        pm.sprint('run name: {}'.format(pm.run.name),1)
        results = pm.results

        # Execute required jobs
        results.add(SPiDEA.main(pm), name='ext')

        # All jobs done
        if pm.run.save:
            # store log in file
            f = open(pm.output_dir + '/SPiDEA.log', 'w')
            f.write(pm.log)
            f.close()

            # need to get rid of nested functions as they can't be pickled
            tmp = copy.deepcopy(pm)
            del tmp.sys.v_ext
            del tmp.sys.v_pert

            # store pickled version of parameters object
            import pickle
            f = open(pm.output_dir + '/parameters.p', 'wb')
            pickle.dump(tmp, f, protocol=4)
            f.close()

            del tmp

        results.log = pm.log
        pm.log = ''  # avoid appending, when pm is run again

        string = 'all jobs done \n'
        pm.sprint(string,1)

        return results
