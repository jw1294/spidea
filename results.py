"""Bundles and saves iDEA results

"""
import copy as cp
import numpy as np
import pickle


class Results(object):
    """Container for results.

    A convenient container for storing, reading and saving the results of a
    calculation.

    Usage::

      res = Results()
      res.add(my_result, 'my_name')
      res.my_name  # now contains my_result
      res.save(pm)  # saves to disk + keeps track

      res.add(my_result2, 'my_name2')
      res.save(pm)  # saves only my_result2 to disk

    """

    calc_dict = {
        'td': 'time-dependent',
        'gs': 'ground state',
    }
    method_dict = {
        'ext': 'exact',
    }
    quantity_dict = {
        'den': r'$\rho$',
        'vxt': r'$V_{ext}$',
        'tden' : r'$\rho$',
    }


    def __init__(self):
        self._saved = set()  # list of results already saved to disk


    @property
    def _not_saved(self):
        """Returns list of results not yet saved to disk."""
        results_names = [ r for r in self.__dict__.keys() if not r.startswith('_') ]
        return [r for r in results_names if r not in self._saved]

    @staticmethod
    def label(shortname):
        r""" returns full label for shortname of result.

        Expand shortname used for quantities saved by iDEA.
        E.g. 'gs_non_den' => 'ground state $\rho$ (non-interacting)'
        """
        c, m, q = shortname.split('_')
        s  = r"{} {} ({})".format(Results.calc_dict[c], Results.quantity_dict[q], Results.method_dict[m])
        return s


    def add(self, results, name):
        """Add results to the container.

        Note: Existing results are overwritten.
        """
        # if results of same name was saved already,
        # ask to save again
        if hasattr(self, name) and name in self._saved:
            self._saved.remove(name)
        # if name exists and we are adding another Results instance,
        # copy its attributes
        if hasattr(self, name) and isinstance(results, Results):
            getattr(self, name).__dict__.update(results.__dict__)
        # else, we simply deepcopy the results
        else:
            setattr(self, name, cp.deepcopy(results))


    @staticmethod
    def read(name, pm, dir=None):
        """Reads and returns results from pickle file

        parameters
        ----------
        name : string
            name of result to be read (filepath = raw/name.db)
        pm : object
            iDEA.input.Input object
        dir : string
            directory where result is stored
            default: pm.output_dir + '/raw'

        Returns data
        """
        if dir is None:
            dir = pm.output_dir + '/raw'
        filename = "{}/{}.db".format(dir,name)
        pm.sprint("Reading {} from {}".format(name,filename),0)
        #pm.sprint("Reading {} from {}".format(Results.label(name),filename),0)
        f = open(filename, 'rb')
        data = pickle.load(f, encoding='latin1')
        f.close()
        return data


    def add_pickled_data(self, name, pm, dir=None):
        """Read results from pickle file and adds to results.

        parameters
        ----------
        name : string
            name of results to be read (filepath = raw/name.db)
        pm : object
            iDEA.input.Input object
        dir : string
            directory where result is stored
            default: pm.output_dir + '/raw'
        """
        data = self.read(name, pm, dir)
        self.add(data, name)


    def save(self, pm, dir=None, list=None):
        """Save results to disk.

        Note: Saves only results that haven't been saved before.

        parameters
        ----------
        pm : object
            iDEA.input.Input object
        dir : string
            directory where to save results
            default: pm.output_dir + '/raw'
        verbosity : string
            additional info will be printed for verbosity 'high'
        list : array_like
            if given, saves listed results
            if not set, saves results that haven't been saved before
        """
        if dir is None:
            dir = pm.output_dir + '/raw'

        if list:
            to_save = list
        else:
            to_save = self._not_saved

        for key,val in self.__dict__.items():
            if key in to_save:
                if isinstance(val,Results):
                    val.save(pm, dir)
                else:
                    outname = "{}/{}.db".format(dir,key)
                    pm.sprint("Saving {} to {}".format(key,outname),0)
                    f = open(outname, 'wb')
                    pickle.dump(val,f,protocol=4) # protocol=4 for large files (<4GB)
                    f.close()
                    #np.savetxt(outname, val)
                if key not in self._saved:
                    self._saved.add(key)
