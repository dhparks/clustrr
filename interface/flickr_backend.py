### backend for the flickr stats demo
from __future__ import print_function
from __future__ import absolute_import

DEBUG = False

# web service apis
if not DEBUG:
    from flickrapi import FlickrAPI
    from . import flickr_help as fh
    
from . import api_keys as ak
from . import tag_classes as tc
from . import som_classes as sc

import numpy as np
import math
from PIL import Image
import os

class Backend(object):
    
    """ This class defines an api that the gui can talk to through the browser,
    interfacing between the browser commands and the database commands
    in tag_classes and the analytical routines in som_classes.
   
    Most functions in this class get called in an indirect fashion.
    The server file calls backend.cmds[cmd](*args). Cmds is a dictionary
    linking browswer command requests to functions in this class. """

    def __init__(self):

        # start flickr sessions. flickrQ takes care of requests
        if not DEBUG:
            self.flickr = FlickrAPI(ak.flickr_key, ak.flickr_secret)
            self.flickrq = fh.FlickrQ(session=self.flickr)

        # database information
        self.db_root = 'data'

        # data storage objects
        self.cloud = tc.tag_cloud()
        
        self.nsid = None
        self.username = None
        
        # link commands in frontend to commands here in backend
        self.cmds = {'getuser':self._user,
                     'scrape':self._scrape,
                     'buildnetwork':self._build_network,
                     'eigenvalues':self._eigenvalues,
                     'trainsom':self._train,
                     'assignments':self._clustering,
                     'centralities':self._centralities,
                     'download':self._download}
        
        self.fail = {'status':'fail', 'message':'unknown error'}
        self.success = {'status':'ok'}

        # other data structures to be populated later
        self.graph = tc.tag_graph()
        self.matrix = tc.tag_matrix()
        self.som = sc.MultiSOM((15, 15))
        
        # for file saving
        self.file_id = 'user_%s maxdate_*'
        
        # special switches for looking at photos in explore
        self.popular_tags = 200
        self.remove_vision_tags = False
        
    def get_uid(self, url):
        """ Get the user id from the flickr url of form
        http://flickr.com/photos/$username
        This sets the userid for the rest of the backend.
        
        requires: url
        """

        # todo: validate URL!!!
        if not DEBUG:
            self.nsid, self.uname = self.flickrq.user_from_url(url)
        else:
            self.nsid = '59378287@N03'
            self.uname = 'safanda'
            #self.nsid = ak.flickr_me
            #self.uname = 'D. H. Parks'
            
            
        self.db_path = os.path.join(self.db_root, self.nsid)
        try:
            os.makedirs(self.db_path)
        except OSError:
            pass
        
    def special_name_explore(self):
        
        """ Set special values for getting data from explore """
        
        self.nsid = 'explore'
        self.uname = 'explore'
        self.db_path = os.path.join(self.db_root, self.nsid)
        self.popular_tags = 400
        try:
            os.makedirs(self.db_path)
        except OSError:
            pass
        
    def connect(self, nsid=None):
        """ Switch the database being interrogated or populated
        by this instance of the backend. The optional parameter
        nsid will set the flickr nsid.
        
        Requires: None
        Optional: nsid, a flickr userid like 
        """
        
        if nsid != None:
            self.nsid = nsid
        
        # first, close the currently open database
        try:
            self.cloud.database.close()
        except (NameError, AttributeError):
            pass
        
        # connect or create using the usual tag schema.
        # see tag_classes.tag_cloud for details
        self.cloud.connect(os.path.join(self.db_path, '%s.db'%self.nsid))
        self.cloud.add_tables()
       
    def populate_explore(self):
        """ Fill the database with photos from Flickr's EXPLORE """

        if not DEBUG:

            # check the date of the most recent photo.
            try:
                    
                # add new information to database
                fields = self.cloud.post_fields
                extras = ','.join(fields)
                
                start = '2013-01-01'
                end = '2013-01-30'
                have_days = self.cloud.all_dates()
                
                for yielded in self.flickrq.explore_photos(fields=fields, extras=extras, start=start, end=end, have_days=have_days):
                    self.cloud.add_photos(yielded)

            except fh.FlickrError as err:
                print(err.args[0]['message'])
                raise
        
        else:
            pass
        
    def populate(self):
        """ Fill the database with the information about
        the user's photos. """

        # skip querying flickr
        if not DEBUG:
    
            # get a stop date for the photos
            max_date = self.cloud.date_min_max()[1]
    
            # check the date of the most recent photo.
            try:
                if self.flickrq.check_date(nsid=self.nsid, stop_date=max_date):
                    
                    # add new information to database
                    fields = self.cloud.post_fields
                    extras = ','.join(fields)
                    photos = self.flickrq.user_photos(nsid=self.nsid, fields=fields, extras=extras, stop_date=max_date)

                    self.cloud.add_photos(photos)
                    
                    # remove old analysis
                    import glob
                    max_date = self.cloud.date_min_max()[1]
                    files_in_directory = glob.glob(os.path.join(self.db_path, '*'))
                    for fname in files_in_directory:
                        try:
                            f_max_date = int(fname.split('maxdate_')[1].split('.')[0])
                            if f_max_date < max_date:
                                os.remove(fname)
                            else:
                                pass
                        except IndexError:
                            pass

            except fh.FlickrError as err:
                print("received error")
                print(err.args[0]['message'])
                raise
        
        else:
            pass
        
    def _load_old(self, fmt, load, calculate, after=None):
        """ Helper function which tries to load old analysis.
        If the analysis is out of date, calculates new analysis
        and saves it.
        
        fmt: format of old analysis
        fn: filename for new analysis
        load: loading function
        calculate: calculation and save function
        after: things to do after loading/calculating
        """
        
        from glob import glob
        
        fname = fmt.replace('*', '%s')
        
        # get the max_date photo in the database
        max_date = self.cloud.date_min_max()[1]
       
        # find all the old files for this userid
        files = glob(os.path.join(self.db_path, fmt%self.nsid))
        files.sort()

        # check for old work. if it exists, open it. if it is out
        # of date, recalculate it.
        if len(files) > 0:
            file_date = float(files[-1].split('maxdate_')[1].split(fmt.split('.')[-1])[0])
            if file_date == max_date:
                load(files[-1])
            else:
                calculate(os.path.join(self.db_path, fname%(self.nsid, max_date)))
        else:
            calculate(os.path.join(self.db_path, fname%(self.nsid, max_date)))
        
        if after != None:
            after()

    def _user(self, args, json, form):
        """ Get the user id based on the photo page url """
        try:

            name = form['name']

            # special names:
            if name == '$explore':
                self.special_name_explore()
            else:
                self.get_uid('http://flickr.com/photos/%s'%form['name'])
            return {'status':'ok', 'nsid':self.nsid}
        except fh.FlickrError as err:
            return {'status':'fail', 'message':err.args[0]['message']}

    def _scrape(self, args, json, form):
        # connect to the database for the user
        # download user's photo info
        
        """ Delegates to _populate() or _populate_explore()
        depending on the user name """

        try:
            self.connect()
        except:
            return self.fail
        
        try:
            if self.uname == 'explore':
                self.populate_explore()
            else:
                self.populate()
        except fh.FlickrError as err:
            return {'status':'fail', 'message':err.args[0]['message']}

        return self.success
    
    def _centralities(self, args, json, form):
        
        """ Calculate the centralities of the various clustered
        subgraphs """
        
        import json as j
        
        def _calculate(fname):
            """ Calculation/saving helper """
            self.centralities = {str(key):self.graph.calculate_centralities(val)
                                 for key, val in self.cluster_members.items()}
            with open(fname, 'w') as where:
                j.dump(self.centralities, where)
        
        def _load(fpath):
            """ Load old results helper """
            with open(fpath, 'r') as where:
                try:
                    self.centralities = j.load(where)
                except:
                    _calculate(fpath)
                
        try:
            fmt = 'centralities %s.json'%self.file_id
            self._load_old(fmt, _load, _calculate)
            return self.success
        except:
            return self.fail
    
    def _build_network(self, args, json, form):
        """ Turn the bag of words scraped by _scrape and
        _populate into a networkx graph """
        
        
        def _calculate(fname):
            """ Calculate/save helper """
            from networkx import write_gpickle
            self.graph.source = self.cloud
            self.graph.make_from_bags(popular_tags=self.popular_tags,
                                      remove_vision_tags=self.remove_vision_tags)
            write_gpickle(self.graph, fname)
            
        def _load(fpath):
            """ Load old results helper """
            from networkx import read_gpickle
            self.graph = read_gpickle(fpath)
            
        def _after():
            """ Do things afterwards helper """
            self.matrix.source = self.graph
            self.matrix.make_matrix()
            
        #try:
        fmt = 'graph user_%s maxdate_*.pck'
        self._load_old(fmt, _load, _calculate, after=_after)
        return self.success
        #except:
            #return self.fail
    
    def _eigenvalues(self, args, json, form):
        """ Calculate eigenvalues of the counts matrix for cluster
        size determination."""
        try:
            self.matrix.eigenvalues(component='counts')
            return self.success
        except:
            return self.fail

    def _train(self, args, json, form):
        """ Train the self-organizing map many times, then look
        for repeats. """
        
        def _calculate(fname):
            """ Calculate/save helper """
            training_set = self.matrix.fuzzy
            self.som.graph = self.graph
            self.som.train(training_set, training_set.shape[0]*5, repeats=32, debug=False)
            np.savez_compressed(fname, data=self.som.repeats)
            
        def _load(fname):
            """ Load old results helper """
            self.som.repeats = np.load(fname)['data']
            self.som.samples = self.matrix.fuzzy
            self.som.iterations = self.matrix.fuzzy.shape[0]*5

        try:
            fmt = 'repeat user_%s maxdate_*.npz'
            self._load_old(fmt, _load, _calculate)
            return self.success
        except:
            return self.fail

    def _clustering(self, args, json, form):
        
        """ Calculate cluster reproducibility """
            
        import json as j
        
        def _calculate1(fname):
            """ Helper: calculate reproducibility """
            self.som.cluster_reproducibility()#labels=sorted(self.graph.graph.nodes()))
            with open(fname, 'w') as where:
                j.dump(self.som.reproduction_analysis, where)

            # save the matrices in a compressed format. this helps
            # reduce file size by about 10x due to the highly
            # repetitive nature of the data.
            fname2 = fname.replace('json', 'npz')
            np.savez_compressed(fname2, data=self.som.reproduction_matrices)

        def _calculate2(fname):
            """ Helper: assign prototypes to clusters """
            self.som.assign_prototypes_knn()#node_labels=sorted(self.graph.graph.nodes()))
            with open(fname, 'w') as where:
                j.dump({'clusters':self.som.knn_clusters,
                        'borders':self.som.knn_borders,
                        'umatrix':self.som.umatrixlist}, where)
                
        def _calculate3(fname):
            """ Helper: reformat membership for frontend"""
            self.cluster_members = {str(cluster):tags
                                    for analysis in self.som.reproduction_analysis
                                    for cluster, tags in analysis['members'].items()}
            with open(fname, 'w') as where:
                j.dump(self.cluster_members, where)
                
        def _calculate4(fname):
            """ Helper: calculate modularity of each cluster """
            
            # this is where clusters go to which nclusters designation
            x = {n+1:[str((n**2+n)/2+x) for x in range(n+1)] for n in range(len(self.som.knn_clusters))}

            # now turn this into a list of tags
            nodes = sorted(self.graph.graph.nodes())
            for_modularity = {}
            for y in x:
                tmp = [self.cluster_members[n] for n in x[y]]
                for_modularity[y] = tmp
                
            self.modularity = self.graph.modularity(for_modularity)

            #with open(fname, 'w') as where:
            #    j.dump(self.modularity, where)
            

        def _load1(fname):
            """ Helper: load old results""" 
            with open(fname, 'r') as where:
                self.som.reproduction_analysis = j.load(where)
            loaded = np.load(fname.replace('.json', '.npz'))
            self.som.reproduction_matrices = loaded['data']
                
        def _load2(fname):
            """ Helper: load old results """
            with open(fname, 'r') as where:
                loaded = j.load(where)
                self.som.knn_clusters = loaded['clusters']
                self.som.knn_borders = loaded['borders']
                self.som.umatrixlist = loaded['umatrix']
                self.som.umatrix = np.array(self.som.umatrixlist)
                
        def _load3(fname):
            """ Helper: load old results """
            with open(fname, 'r') as where:
                self.cluster_members = j.load(where)
                
        def _load4(fname):
            """ Helper: load old results """
            with open(fname, 'r') as where:
                self.modularity = j.load(where)

        #try:
        sequence = [{'fmt':'reproduction %s.json'%self.file_id,
                     'load':_load1,
                     'calc':_calculate1},
            
                    {'fmt':'assignments %s.json'%self.file_id,
                     'load':_load2,
                     'calc':_calculate2},
                    
                    {'fmt':'clustermembers %s.json'%self.file_id,
                     'load':_load3,
                     'calc':_calculate3},
                    
                    #{'fmt':'modularities %s.json'%self.file_id,
                    # 'load':_load4,
                    # 'calc':_calculate4}]
                    
                   ]
        
        for seq in sequence:
            self._load_old(seq['fmt'], seq['load'], seq['calc'])
        
        return self.success
        
        #except:
        #    return self.fail
    
    def _download(self, args, json_in, form):
        
        #""" Last stage in the analysis: package data and send it back """
        
        #try:
            
            # assemble data to be downloaded as a dictionary
            to_return = {'status':'ok'}
            
            # for the eigenvalue plot, we just need the eigenvalues.
            # rounding to a specified precision doesn't save much
            # space after compression, so leave at full floating point.
            eigs = self.matrix.eigs.tolist()[:50]#int(len(self.som.repeats)/8)]
            to_return['eigenvalues'] = eigs
            
            # for the som map, we need the umatrix, the borders,
            # and the cluster assignments. 140k raw, 16k zipped.
            to_return['umatrix'] = self.som.umatrixlist
            to_return['borders'] = self.som.knn_borders
            to_return['clusters'] = self.som.knn_clusters
            
            # for the spectral graph, we need to convert to png
            # and supply the url, as well as the number of tags (for scaling)
            max_date = self.cloud.date_min_max()[1]
            to_return['blockdiagonal'] = [s['sizes'] for s in self.som.reproduction_analysis]
            to_return['blockdiagonalurl'] = 'static/images/clusters_%s_%s.png'%(self.uname, max_date)
            to_return['ntags'] = self.graph.graph.number_of_nodes()
            #if not isfile(to_return['blockdiagonalurl']):
            self._to_sprite(self.som.reproduction_matrices, to_return['blockdiagonalurl'])
            
            # stuff for the word cloud
            to_return['tags'] = sorted(self.graph.graph.nodes())
            to_return['members'] = self.cluster_members #11k zipped
            to_return['centralities'] = self.centralities #21k zipped
            to_return['counts'], to_return['maxCounts'] = self.graph.get_counts(tags=to_return['tags'])
            
            # stuff for generating image thumbnails from the wordcloud.
            # need: list of photos each popular tag belong to.
            reverse_table = self.cloud.reverse_table(self.graph.graph.nodes())
            to_return['tagsToPhotos'] = reverse_table['tags_to_photos']
            to_return['photosToThumbs'] = reverse_table['photos_to_thumbs']
            to_return['photosToLinks'] = reverse_table['photos_to_links']

            return to_return
            
        #except:
        #    return self.fail

    def _to_sprite(self, images, name):
        """ save the frames of array as a single large image which only requires a single GET
        request to the webserver. g is the dimensions of the grid in terms of number of images.
        array can be either a np array, in which case all frames are converted
        to images, or an iterable of PIL objects """
        
        import scipy.misc as smp

        def _gridy_diff(nimg, gridx):
            """ Helper """
            gridy = nimg/gridx
            if nimg%gridx != 0:
                gridy += 1
            diff = gridx*gridy-nimg
            return (gridx, gridy, diff)
        
        if isinstance(images, np.ndarray):
            images = [smp.toimage(i) for i in images]

        imgx, imgy = images[0].size
    
        # calculate the best row/column division to minimize wasted space. be efficient
        # with the transmitted bits!
        grid0 = int(math.floor(math.sqrt(len(images)))+1)
        g_list = [_gridy_diff(len(images), gridx) for gridx in [grid0+x for x in range(5)]]
    
        g_list.sort(key=lambda x: x[2])
        gridx = g_list[0][0]
        gridy = g_list[0][1]
    
        big_image = Image.new('L', (int(gridx*imgx), int(gridy*imgy)))
        for nimg, img in enumerate(images):
            idx = int(imgx*(nimg%gridx))
            idy = int(imgy*(nimg//gridx))
            big_image.paste(img, (idx, idy))
        big_image.save(name)
        