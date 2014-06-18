from __future__ import print_function
from __future__ import absolute_import
import sqlite3

def connect(name,factory=None,recreate=False):
    
    """ Connect to the specified database. If the database doesn't exist yet,
    this will create it.
    
    Inputs:
        name -- path to the database file
        factory -- (optional) row factory, default sqlite.Row
        recreate -- (optional) if True, will remove the current file and make
            a new one. This is useful if the old database is corrupted or if
            it has been created by a program under development.
            
    Returns:
        database: the database object
        cursor: a cursor connected to database
    """
    
    import os
    
    from os.path import isfile as isf
    
    if factory == None: factor = sqlite3.Row
    
    if recreate:
        import os
        try: os.remove(dbase)
        except OSError: pass

    database = sqlite3.connect(dbase)
    database.row_factory = sqlite3.Row # make returned queries act more like tuples
    cursor   = database.cursor()
    
    return database, cursor

def add_fts_table(cursor,name,fields):
    """ Add a virtual table to the database using fts4.
    
    Inputs:
        cursor - sqlite cursor object
        name  - name of table
        fields - name of columns
        
    command executed is cursor.execute('create virtual table %name using fts4(%fields)) """
    
    if isinstance(fields,(tuple,list)):
        fields = ','.join(fields)
    
    cursor.execute('create virtual table %s using fts4(%s)'%(name, fields))

def add_tables(cursor,tables):

    """ Add tables to the database linked to cursor.
    
    Inputs:
        cursor -- a sqlite cursor object
        tables -- the tables to add to the database. This should be a dictionary
            whose keys are the names of the table and whose values are a list
            of 2-tuples formatted like [(name, type), (name, type), ... ].
        
    Returns:
        nothing
    """
    
    def _check_types():
        assert isinstance(tables,dict), "tables must be a dictionary"
        for key in tables.keys():
            assert isinstance(tables[key],(tuple,list))
            for entry in tables[key]:
                assert isinstance(entry,(tuple,list)) and len(entry) == 2, "entry %s incorrectly formatted"%entry

    _check_types()
    
    for name in tables.keys():
        try:
            cmd = 'create table %s (%s)'%(name,','.join(['%s %s'%(c[0],c[1]) for c in tables[name]]))
            cursor.execute(cmd)
        except sqlite3.OperationalError:
            pass
