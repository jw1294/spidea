# import iDEA Input class
from input import Input

# read parameters file into Input object
pm = Input.from_python_file('parameters.py')

# perform checks on input parameters
pm.check()

# run job
pm.execute()
