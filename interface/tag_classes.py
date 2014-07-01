from __future__ import print_function
from __future__ import absolute_import

import sqlite3
from . import db_help as dbh
import networkx as nx
import time
import numpy as np

class DefaultParser(object):
    
    """ Default parsing class for TagCloud """
    
    def __init__(self, stop_words='commonWords.txt'):
        if stop_words != None:
            s_words = open(stop_words)
            self.stop_words = set(s_words.read().split('\n'))
            s_words.close()
        else:
            self.stop_words = None

        # potential future use for these; not used now
        self.brand_tags = ('nikon', 'olympus', 'sony', 'canon',
                           'iphone', 'tamron', 'sigma', 'd5100')

    def parse(self, entries, remove_vision_tags=False):
        """ Parses a string_tuple coming out of the database.
        coming from flickr, the title, tags, and description
        fields have had some minimum processing already applied:
        
        1. html is stripped in the description field
        2. newline characters (\n) are replaced with spaces (' ') in the
        description field
        
        the default parser will
        1. combine the elements of the tuple
        2. make lower case
        3. strip punctuation (except for hastags #)
        4. split the resulting string into tokens
        5. remove stop words
        
        a different parser could be created to remove things like technical tags
        or flick's mysterious vision:$something tags (computer vision research?)
        """
        
        def _helper(entry):
            """ Helper """
            text = (' '.join(entry)).lower()
            text = set(text.split(' '))
            if self.stop_words != None:
                text = text.difference(self.stop_words)
            if remove_vision_tags:
                text = set([x for x in text if 'vision_' not in x])
            return text
        
        return [_helper(x) for x in entries]

class TagCloud(object):
    """ This class manages the tag cloud, providing methods for adding posts
    and tags, and for some analysis of the tagcloud. This class does not
    directly talk to the photo services like flickr or instagram.
    """
    
    def __init__(self):
        
        self.database = None
        self.cursor = None
        self.total_tags = 0
        self.source = "flickr"
        self.source_string = "flickr"

        # intial attribute declarations
        self.fts_fields = None
        self.post_fields_string = None
        self.post_fields = None
        self.post_fields_set = None
        self.tables = None

    def connect(self, name, factory=None, recreate=False):
    
        """ Connect to the specified database. If the database doesn't exist
        yet, this will create it.
        
        Inputs:
            name -- path to the database file
            factory -- (optional) row factory, default sqlite.Row
            recreate -- (optional) if True, will remove the current file and
                make a new one. This is useful if the old database is corrupted
                or if it has been created by a program under development.
                
        Returns:
            database: the database object
            cursor: a cursor connected to database
        """
        
        import os

        if recreate and name != ":memory:":
            try:
                print("removing for re-creation")
                os.remove(name)
            except OSError:
                print("error removing %s"%name)
    
        # connect
        database = sqlite3.connect(name)
        cursor = database.cursor()
        
        # apply row factory
        if factory == None:
            factory = sqlite3.Row
        database.row_factory = factory
        
        # add to namespace
        self.database = database
        self.cursor = cursor
     
    def add_tables(self):
    
        """ Add the 3 tables used to manage photo-tags through SQL to the
        database linked with cursor.
        
        Returns:
            nothing, but the database now has the correct 3 tables"""
        
        if self.database == None:
            print("need to connect to database before adding tables")
            
        else:
            # make a list of the fields to put into table posts.
            # primary key is reserved
            texts = ['source', 'title', 'description', 'tags',
                     'geo', 'owner', 'thumbnail', 'o_dims', 'url']
            
            ints = ['date_taken', 'date_upload']
            
            post_fields = [('id', 'text primary key'),]+\
                            [(x, 'text') for x in texts]+\
                            [(x, 'integer') for x in ints]
            
            fts_fields = ('id', 'owner', 'title', 'description',
                          'tags', 'thumbnail', 'url')

            self.post_fields = [g[0] for g in post_fields if g != 'source']
            self.post_fields_set = set(self.post_fields+['source',])
            self.post_fields_string = ','.join(self.post_fields)
            self.fts_fields = set(fts_fields)
            
            try:
                # make the schema
                self.tables = {}
                self.tables['posts'] = post_fields
                self.tables['clusters'] = (('word', 'text'), \
                    ('cluster', 'integer'))
                
                # add the tables to the database
                dbh.add_tables(self.cursor, self.tables)
                dbh.add_fts_table(self.cursor, 'fts_lookup', fts_fields)
                    
                # add the preliminary dummy to tagmap
                self.database.commit()
            except sqlite3.OperationalError:
                pass

    def add_photo(self, photo, defer_commit=True):
        """ Add a photo to the database. """
        
        def _check(photo):
            """ Check to see if a photo is new and has semantic content """
        
            def _photo_is_new(pid):
                """ Helper: see if photo is new """
                self.cursor.execute('select * from posts where id=?', (pid,))
                return None == self.cursor.fetchone()
            
            pid = photo['id']
            if _photo_is_new(pid):
                test = photo['tags'] != ('')
                test = test or photo['description'] != ('')
                test = test or photo['title'] != ('')
                if test:
                    return True
            return False

        def _insert_post():
            """ Insert the post information into the database tables """
            
            for key in photo.keys():
                # we should never trigger this assertion; the parser
                # in flickr_help has already verified the fields
                assert key in self.post_fields_set
            
            fields = ','.join([field for field in photo])
            values = [photo[field] for field in photo]
            
            fts_fields = ','.join([f for f in photo if f in self.fts_fields])
            fts_values = [photo[f] for f in photo if f in self.fts_fields]

            joined1 = ','.join(len(values)*['?',])
            joined2 = ','.join(len(fts_values)*['?',])
            q_base = 'insert or ignore into %s (%s) values (%s)'
            query1 = q_base%('posts', fields, joined1)
            query2 = q_base%('fts_lookup', fts_fields, joined2)

            self.cursor.execute(query1, values)
            self.cursor.execute(query2, fts_values)

        if _check(photo):
            
            # append the source field which is NOT part of the web response
            photo['source'] = self.source_string
            _insert_post()
        
            if not defer_commit:
                # usually, we add multiple photos and defer the commit until
                # all inserts are finished.
                self.database.commit()
            
    def add_photos(self, photos):
        """ Add a bunch of photos to the database """
        for photo in photos:
            self.add_photo(photo)
        self.database.commit()

    def date_min_max(self):
        """ Get min and max upload date for this database """
        query = 'select min(date_upload), max(date_upload) from posts'
        
        return self.cursor.execute(query).fetchone()

    def all_dates(self):
        """ Get the number of different dates in this database.
        For explore photos """
        
        query = 'select distinct date_upload from posts'
        return [x[0] for x  in self.cursor.execute(query).fetchall()]

    def reverse_table(self, popular_tags):
        """ Given a list of popular_tags, return several dictionaries:
        
        1. tags_to_photos {tag1:{pid1:None, pid2: None}, tag2: {}....}
        2. photos_to_thumbs {pid1: thumbnail_url1, pid2: thumbnail_url2,...}
        3. photos_to_links {pid1: href1, pid2: href2, ...}
        
        tags_to_photos is structured as a dictionary of dictionary of Nones
        to allow for fast searching through keys in the javascript front end.
        (hash searching is fast!) """
        
        def _parse(record):
            """ Helper """
            return {'id':record[0], 'thumb':record[1], 'url':record[2]}
        
        query = "select id, thumbnail, url from fts_lookup where tags match ?"

        tags_to_photos, photos_to_thumbs, photos_to_links = {}, {}, {}

        for tag in popular_tags:
            
            # get info from the fts table
            photos = [_parse(x) for x in self.cursor.execute(query, (tag,))]
            
            # process it for the return to the front
            # tags_to_photos is hashed for fast searching in js
            tags_to_photos[tag] = {x['id']:None for x in photos}
            photos_to_thumbs.update({x['id']:x['thumb'] for x in photos})
            photos_to_links.update({x['id']:x['url'] for x in photos})

        return {'tags_to_photos':tags_to_photos,
                'photos_to_thumbs':photos_to_thumbs,
                'photos_to_links':photos_to_links}

class TagGraph(object):
    
    """ This class takes a TagCloud, which is really just an interface to a
    SQLite database, and turns it into a networkX graph whose nodes
    are tokens and whose edges are token correlations. Making the cloud
    into a graph allows use of some graph analysis tools.
    """
    
    def __init__(self, source=None, parser=DefaultParser):
        
        # source is the tagcloud which manages the tag database
        self.source = source
        self.graph = nx.DiGraph() # use digraph for assymmetric affinities
        self.parser = parser()
        #self.get = 'select title, description, tags from posts'
        self.get = 'select owner, tags from posts'
        self.adjmat = None

    def get_counts(self, tags=None):
        """ Get the number of counts for each tag, scaled so that
        the max count is 255 """
        import math
        if tags == None:
            tags = sorted(self.graph.nodes())
        counts = [math.sqrt(self.graph.node[tag]['count']) for tag in tags]
        maxc = max(counts)
        return dict(zip(tags, [int(255*x/maxc) for x in counts])), maxc

    def calculate_centralities(self, indices=None, tags=None):
        """ Calculate the eigenvalue centralities of a sub graph
        defined by indices or tags. Either tags or indices can be
        supplied. If indices is supplied, tags is generated
        by self.graph.nodes()[indices]. If both are supplied,
        only tags is used.
        
        Returns: eigenvector centralities of subgraph. """

        from scipy import linalg

        # make the subgraph from tags
        if tags == None and indices == None:
            return None
        if tags == None and indices != None:
            nodes = sorted(self.graph.nodes())
            tags = [nodes[i] for i in indices]
        if len(tags) <= 2:
            return {tag:0 for tag in tags}
        subgraph = self.graph.subgraph(tags)

        # make into matrix and remove self edges
        data = np.array(nx.to_numpy_matrix(subgraph, weight='count'))
        #data += data.T
        size = data.shape[0]
        for step in range(size):
            data[step, step] = 0
    
        # eigenvector calculation
        value, vector = linalg.eigh(data, eigvals=(size-1, size-1))
        vector = vector.flatten()
        norm = np.sign(vector.sum())*linalg.norm(vector)
        vector /= norm
        vector = (1-np.exp(-2*vector))/(1-np.exp(-2))
        vector = [int(255*x) for x in vector]
        
        centralities = dict(zip(subgraph, vector))

        return centralities

    def make_from_bags(self,
                       metrics=None,
                       min_user_incidence=.001,
                       min_post_incidence=.0001,
                       min_node_count=0,
                       min_edge_count=0,
                       popular_tags=None,
                       remove_vision_tags=False):

        """ Actually build the tag graph out of the tokens contained in the
        semantic fields of the database. Some of the tag-comparison metrics,
        such as the fuzzy logic similarity, are expensive to calculate and so
        we first need to cut down the network's nodes and edges. Here are
        several options to cut down the number of tags:
        
            1. N most popular tags. If popular_tags is specified with an
            integer, only that number of popular words is retained.
            
            2. min_node_count/min_edge_count: If given an integer, only nodes
            with at least this many occurences survive the cut. Similarly for
            min_edge_count, where the count of and edge is the number of times
            two tags/words co-occur.
            
            3. min_post_incidence/min_user_incidence: these is a numerical
            value between 0 and 1 which indicate in what percentage of photos
            a token must be present (min_post_incidence) for the photo to
            survive the cut. Similarly for min_user_incidence, this fraction of
            unique users in the dataset must have used the tag; userful for
            cutting tags unique to a single person or a small clique.
        
        """
        
        def _nodes():
            """ Helper """
            return self.graph.nodes_iter(data=True)
        
        def _edges():
            """ Helper """
            return self.graph.edges_iter(data=True)
        
        def _add_nodes():
            """ Subfunction which adds the nodes to the graph,
            tracking node counts """
            # add the nodes. each time we see a term, increment the node count.
            # also maintain a SET of the different users who have used each tag;
            # this helps eliminate tags used by just a few people.
            self.node_users = {}
            for user, entry in zip(users, terms): 
                for term in entry:
                    try:
                        self.graph.node[term]['count'] += 1
                    except KeyError:
                        self.graph.add_node(term, count=1, nuser=1)
                    try:
                        self.node_users[term].add(user)
                    except KeyError:
                        self.node_users[term] = set([user,])
                    
            try:    
                for node in self.graph.nodes_iter():
                    self.graph.node[node]['nusers'] = len(self.node_users[node])
            except KeyError:
                print("key error nusers! %s"%node)
                pass
            
        def _remove_rare_tags():
            """ Subfunction which removes the rare tags """
            # remove rare tags, which are either semi-unique to users
            # or are semi-unique to posts\
            to_remove1 = []
            if min_user_incidence > 0:
                rare1 = int(min_user_incidence*nusers)
                for n, d in _nodes():
                    try:
                        if d['nusers'] < rare1:
                            to_remove1.append(n)
                    except KeyError:
                        pass
                
            rare2 = 0
            if min_post_incidence > 0: 
                rare2 = int(min_post_incidence*nposts)
                
            if min_node_count > 0:
                rare2 = min_node_count
            
            if rare2 > 0:
                to_remove2 = [n for n, d in _nodes() if d['count'] < rare2]
            else:
                to_remove2 = []
                
            self.graph.remove_nodes_from(set(to_remove1).union(set(to_remove2)))
            
        def _most_popular_tags():
            """ Subfunction which deletes all but the most popular tags """
            # keep only the N-most popular tags.
            try:
                counts = [x for x in _nodes()]
                counts.sort(key=lambda x: x[1]['count'])
                to_remove = [x[0] for x in counts[:-popular_tags]]
                self.graph.remove_nodes_from(to_remove)
            except (TypeError, IndexError):
                pass
                
        def _add_edges():
            """ Subfunction which adds edges between nodes of the graph """
            # use itertools to efficiently loop over all the unique token pairs.
            # this is O(N**2) so it runs pretty slowly in python.
            # for each edge, we increment the count of the edge.
            from itertools import combinations
            for entry in terms:
                entry = set(entry)
                entry = entry.intersection(remaining_nodes)
                for word0, word1 in combinations(entry, 2):
                    try:
                        self.graph[word0][word1]['count'] += 1
                    except KeyError:
                        self.graph.add_edge(word0, word1, count=1)
                    try:
                        self.graph[word1][word0]['count'] += 1
                    except KeyError:
                        self.graph.add_edge(word1, word0, count=1)        
        
        def _remove_rare_edges():
            """ Subfunction that removes rare edges """
            # remove rare tag co-occurrences
            if min_edge_count > 0:
                to_remove = [(w1, w2) for w1, w2, d in _edges() \
                    if d['count'] < min_edge_count]
                self.graph.remove_edges_from(to_remove)
        
        def _edge_counts():
            """ precompute edge count sums for faster performance"""
            edge_count_sums = {}
            for node in self.graph.nodes_iter():
                gnode = self.graph[node]
                summed = sum([gnode[n].get('count', 0) for n in gnode.keys()])
                edge_count_sums[node] = summed
            return edge_count_sums
                
        def _calculate_edge_metrics():
            """ Calculate pair-wise edge metrics; current jaccard and
            fuzzy_logic """
            
            def _min(gw0, gw1, m):
                """ Helper """ 
                return min((gw0[m]['count'], gw1[m]['count']))
            
            for word0, word1, data in _edges():
    
                edge0 = self.graph[word0][word1]
                edge1 = self.graph[word1][word0]
                
                if 'jaccard' in metrics and 'jaccard' not in edge1:
                
                    #jaccard similiarity (size of intersection/size of union)
                    node_count0 = self.graph.node[word0]['count']
                    node_count1 = self.graph.node[word1]['count']
                    edge_count = data['count']
                    j = edge_count*1./(node_count0+node_count1-edge_count)
                    edge0['jaccard'] = j
                    edge1['jaccard'] = j
                    
                b_tmp = ('fuzzylogic' not in edge1 or 'fuzzylogic' not in edge0)
                if 'fuzzylogic' in metrics and b_tmp:
                
                    # calculate fuzzy logic similiarity. computationally more
                    # expensive, as we have to iterate over additional edges.
                    gw0 = self.graph[word0]
                    keys0 = gw0.keys()
                    
                    gw1 = self.graph[word1]
                    keys1 = gw1.keys()
                    
                    ikeys = set(keys0).intersection(set(keys1))
                    mins = [_min(gw0, gw1, m) for m in ikeys]
                    summed = sum(mins)*1.
                    
                    if 'fuzzylogic' not in edge0:
                        edge0['fuzzylogic'] = summed/edge_count_sums[word0]
                    if 'fuzzylogic' not in edge1:
                        edge1['fuzzylogic'] = summed/edge_count_sums[word1]
            
            for word in self.graph.nodes():
                for met in metrics:
                    eval('self.graph.add_edge(word, word, %s=1)'%met)

        if metrics == None:
            metrics = ['jaccard', 'fuzzy_logic']

        # first, add the nodes to the graph by splitting up each bag of words
        # returned: a list of sets. each set contains tokens for the graph
        c_exec = self.source.cursor.execute
        nposts = c_exec('select count(id) from posts').fetchone()[0]
        entries = c_exec(self.get).fetchall()
        users = [e[0] for e in entries]
        nusers = len(set(users))
        terms = self.parser.parse([e[1:] for e in entries], \
            remove_vision_tags=remove_vision_tags)
        
        n = self.graph.number_of_nodes()

        # add all the tags as nodes, then remove those that are rare. keep
        # only those that are common
        _add_nodes()
        _remove_rare_tags()
        _most_popular_tags()
        remaining_nodes = set(self.graph.nodes())
        print('%s remaining nodes'%len(remaining_nodes))

        # now add edges, and remove those that are rare
        _add_edges()
        _remove_rare_edges()

        # calculate edge metrics. currently, these are jaccard and fuzzy_logic.
        edge_count_sums = _edge_counts()
        _calculate_edge_metrics()
        
class TagMatrix(object):
    """Treats the tag cloud as a matrix for things like eigenvalue
    calculation. Currently not very useful."""
    
    def __init__(self, source=None):
        self.source = source

        # initial attribute declarations
        self.fuzzy = None
        self.jaccard = None
        self.count = None
        self.eigs = None
        self.fuzzy_reduced = None
        self.jaccard_reduced = None
        
        # if we already have the source, calculate
        # adjacency matrices
        if source != None:
            self.make_matrix()

    def make_matrix(self, source=None):
        
        """ Make arrays for each of the available metrics.
        Matrices are ordered according to a sorted node list. """
        
        def to_matrix(what):
            """ Helper """
            tmp = nx.to_numpy_matrix(source.graph, weight=what, nodelist=nodes)
            return np.array(tmp)
        
        if source == None:
            source = self.source

        nodes = sorted(source.graph.nodes())

        self.fuzzy = to_matrix('fuzzylogic')
        self.jaccard = to_matrix('jaccard')
        self.counts = to_matrix('count')
        
    def eigenvalues(self, component='counts', power=1):
        """ Calculate the normalized graph laplacian for a matrix component,
        and from the laplacian the eigenvalues """
        
        from scipy import linalg
        data = eval('self.%s'%component)
        
        if component == 'jaccard':
            assert isinstance(power,(int,float)), "power must be number"
        
        data = data**power
        
        dmatrix = 1.*np.eye(data.shape[0], dtype=np.int)*np.sum(data, axis=0)
        dmatrix2 = 1.*np.eye(data.shape[0], dtype=np.int)
        dmatrix2 *= 1./np.sqrt(np.sum(data, axis=0))
        laplace = dmatrix-data
        laplace_sym = np.dot(dmatrix2, np.dot(laplace, dmatrix2))

        # from the symmetric laplacian lsym, calculate the eigenvalues
        eigs = np.abs(linalg.eigvals(laplace_sym).real)
        eigs.sort()

        w, v = linalg.eig(laplace_sym)
        self.eigs = eigs
        
    def svd_reduce(self, components=None):
        """ Use the singular value decomposition to reduce the dimensionality
        of the similarity matrices. For example: 
        
        Required inputs:
            None
            
        Optional inputs:
            components - If an (integer) is supplied, it is the number
                of eigenvecrtors to use in reducing the dimensionality.
                If a (float) is supplied, it is the fraction of the total
                variance captured by the SVD reduction. By default, half
                the singular values are kept, reducing the dimensionality
                by half.
                
        Returns:
            nothing, but creates 2 new matrices:
                self.fuzzy_reduced
                self.jaccard_reduced
        """
        
        def _reduce(matrix):
            """ Helper which performs the actual svd decomposition"""
            
            # decompose
            U, s, Vh = svd(matrix)
            
            # sort
            i = np.argsort(s)[::-1]
            U = U[:, i]
            
            # recompose
            if isinstance(components, float):
                power = np.abs(s) # might be complex
                cumulatives = np.cumsum(power)
                cumulatives /= cumulatives[-1]
                j = 0
                while cumulatives[j] < components:
                    j += 1
                j += 1
                
            if isinstance(components, int):
                j = components

            reduced = np.abs(matrix.dot(U[:, :j]))
            
            return reduced

        # see if this exists. if not, make it
        try:
            self.fuzzy *= 1
        except NameError:
            self.make_matrix()
        
        # set default number of componetns
        if components == None:
            components = self.fuzzy.shape[0]/2
            
        # run the svd/pca
        from scipy.linalg import svd
        self.fuzzy_reduced = _reduce(self.fuzzy)
        self.jaccard_reduced = _reduce(self.jaccard)
            
            
        
        
        
        
        
        
        
        
        