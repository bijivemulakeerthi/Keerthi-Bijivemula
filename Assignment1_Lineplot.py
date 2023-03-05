Python 3.9.13 (main, Aug 25 2022, 18:29:29) 

Type "copyright", "credits" or "license" for more information.



IPython 7.31.1 -- An enhanced Interactive Python.



In [1]: runfile('/Users/apple/Visualisation1.py', wdir='/Users/apple')



WARNING: This is not valid Python code. If you want to use IPython magics, flexible indentation, and prompt removal, we recommend that you save this file with the .ipy extension.



Traceback (most recent call last):



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/spyder_kernels/py3compat.py", line 356, in compat_exec

    exec(code, globals, locals)



  File "/Users/apple/Visualisation1.py", line 14, in <module>

    df = pd.read_csv('/Users/apple/Desktop/Assignment')



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper

    return func(*args, **kwargs)



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 678, in read_csv

    return _read(filepath_or_buffer, kwds)



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 575, in _read

    parser = TextFileReader(filepath_or_buffer, **kwds)



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 932, in __init__

    self._engine = self._make_engine(f, self.engine)



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1216, in _make_engine

    self.handles = get_handle(  # type: ignore[call-overload]



  File "/Users/apple/opt/anaconda3/lib/python3.9/site-packages/pandas/io/common.py", line 786, in get_handle

    handle = open(



IsADirectoryError: [Errno 21] Is a directory: '/Users/apple/Desktop/Assignment'



runfile('/Users/apple/Visualisation1.py', wdir='/Users/apple')



In [2]: runfile('/Users/apple/Visualisation1.py', wdir='/Users/apple')

Index(['id', ' timestamp', ' demand', ' frequency', ' coal', ' nuclear',

       ' ccgt', ' wind', ' pumped', ' hydro', ' biomass', ' solar'],

      dtype='object')


 

Warning

Figures now render in the Plots pane by default. To make them also appear inline in the Console, uncheck "Mute Inline Plotting" under the Plots pane options menu.




In [3]: 