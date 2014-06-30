""" Backend python libraries for Clustrr, a web app which uses
machine learning techniques to identify hierarchical structure
in well-tagged Flickr photostreams.

Author: Daniel Parks (dhparks@lbl.gov)

"""

__version_info__ = ('2014','06','30')
__version__ = '.'.join(__version_info__)

# if you make a new file/module name, put it here.  These are alphabetized.
__all__ = ['interface',]

for mod in __all__:
    exec("import %s" % mod)
del mod
