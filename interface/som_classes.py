from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import itertools
import copy
from scipy import sparse
from multiprocessing import Pool
import networkx

# som training code based on (and dramatically expanded from!):
# http://nbviewer.ipython.org/gist/alexbw/3407544

def find_winner_mp(args):
    """ Implements the find_winner portion of SOM training
    but in a scope accessible by __main__ for multiprocessing."""
    def main(sample, weight, grid):
        """ Helper, does work """
        distances = np.sum((weight-sample)**2, axis=-1)
        return np.unravel_index(distances.argmin(), grid)
    return main(*args)

def find_winners_mp(args):
    
    """ Implements the find_winner portion of SOM training
    but in a scope accessible by __main__ for multiprocessing."""
    
    # args:
    # 0: list of entries (label, index)
    # 1: list of weights
    # 2: grid size
    # 3. samples
    
    # for each each entry, get the right
    # sample. find its winner as a coordinate.
    # generate a winner-tuple as (coords,label)
    
    # return: [(coords,label),(coords,label)]
    
    def main(entries, weights, grid):
        """ Find the winners """
        winners = []
        for label, sample in entries:
            distances = np.sum((weights-sample)**2, axis=-1)
            winners.append((np.unravel_index(distances.argmin(), grid), label))
        return winners
    return main(*args)

def find_winner_mp_sparse(args):
    
    """ Implements the find_winner portion of SOM training
    but in a scope accessible by __main__ for multiprocessing.
    For sparse vectors. """
    
    def main(sample, weight, grid):
        """ Find the winners """
        distances = np.sum((weight-sample)**2, axis=-1)
        return np.unravel_index(distances.argmin(), grid)
    return main(*args)

def _train_mp(args):
    
    """ Train a SOM several times; structured for multiprocessing"""
    
    def _main(grid_size, repeats, samples, iterations, params, graph):
        """ Helper function which actually trains the SOM """
        
        # instantiate the SOM for this CPU
        som = SOM(grid_size)
        
        # build the coocurrence matrix for this cpu
        shape = (2*len(samples),)+samples.shape
        if repeats < 256:
            fmt = np.int16
        else:
            fmt = np.int32
            
        # hashed nodes for fast lookup
        nodes = {x:i for i, x in enumerate(sorted(graph.graph.nodes()))}

        for repeat in range(repeats):
            
            print(repeat)

            # train, agglomerate, assign
            som.train(samples, iterations, params=params)
            som.agglomerator.agglomerate(graph)

            if repeat == 0:
                cluster_repeats = np.zeros(shape, fmt)
            
            for number, clusters in som.agglomerator.cluster_tracker.items():
                # number is the number of clusters
                # clusters is a list/set of clusters
                for cluster in clusters:
                    tags = [nodes[x] for x in som.agglomerator.clusters[cluster].tags]
                    for tag in tags:
                        cluster_repeats[number-1, tag, tags] += 1
                        
        return cluster_repeats

    return _main(*args)

class Cluster(object):
    
    """ Class for cluster objects used in Agglomerator """
    
    def __init__(self, label, umv):
        self.label = label
        self.umv = umv
        self.u_total = 0
        self.coordinates = set([])
        self.tags = set([])
        self.has_tags = False
        
        self.ncoords = 0
        self.ntags = 0
        
        self.tag_members = []
        self.u_average = 0
        
    def add_coord(self, coord):
        """ Add a coordinate on the SOM to this cluster"""
        self.ncoords += 1
        self.coordinates.add(coord)
        
    def add_tag(self, tag):
        """ Add a tag (word, not vector) to this cluster"""
        self.has_tags = True
        self.tags.add(tag)
        self.ntags += 1
        
    def calculate_u_total(self):
        """ Calculate the total value of the umatrix
        locations contained within this cluster """ 
        self.u_total = sum([self.umv[x[0], x[1]] for x in self.coordinates])
        self.u_average = self.u_total/len(self.coordinates)
        
    def merge_with(self, inpt):
        """ Merge clusters to form a new cluster.
        Generally, this will be used as:
        cluster3 = cluster_obj(lable,umv)
        cluster3.merge_with(list_of_other_clusters) """
        
        if isinstance(inpt, Cluster):
            inpt = [inpt,]
        for cluster in inpt:
            self.coordinates = self.coordinates.union(cluster.coordinates)
            self.tags = self.tags.union(cluster.tags)
            
        self.ncoords = len(self.coordinates)
        self.ntags = len(self.tags)
        
        self.calculate_u_total()
        
    def tags_to_indices(self, nodes):
        """ Convert self.tags (which is just a list of indices
        for the input vectors to the SOM) to a list of human-readable
        names generated by querying networkX """

        self.tag_members = [nodes.index(tag) for tag in self.tags]

class Agglomerator(object):
    
    """ Class to hold methods which implement agglomerative
    clustering of the SOM. Any replacement class can be
    substituted provided that it maintains the following
    interface:
    
    1. accepts a SOM class as input on instantiation
    2. accepts a tag-graph class as input on agglomerate
        (this is for checking connectivity)
    3. has attributes self.cluster_tracker and self.clusters,
        where self.cluster_tracker[n] gives the identities of the
        clusters present when n clusters remain to be agglomerated,
        and self.clusters[m] returns a cluster_obj with
        identity m. """
    
    def __init__(self, som, cluster_obj=Cluster):
        
        self.som = som
        self.cluster_obj = cluster_obj
        
        self.clusters = {}
        self.fringe = {}
        self.cluster_tracker = {}
        self.map_tracker = {}
        self.clusters_map = None
        self.current_clusters = None
        
        # graph stuff gets passed at agglomerate()
        self.graph = None
        self.nodes = None

    def agglomerate(self, graph):
        
        """ Perform the hierarchical agglomerative clustering of the
        SOM. Clusters are initially composed of a single activated
        neuron and that neurons nearest neighbors as found by
        sklearn-knn. Through an iterative process, clusters are merged
        until a minimal number of clusters remains. Each iteration,
        only a single pair of clusters is merged; that pair is chosen
        as the pair which minimizes some pair-wise metric, which in this
        case is the total value of the umatrix within each cluster.
        
        Agglomeration using this method requires that the graph of the
        members of the agglomerated cluster be connected.
        
        Inputs:
            graph - the full graph object which generated the feature
                vectors used to train the SOM. This is used to check
                for subgraph connectedness.
                
        Returns:
            None ---BUT--- generates the following results in the self
            namespace:
            
            1. self.clusters: a dictionary where the keys are the cluster
                labels and the values are the corresponding cluster objects.
            2. self.cluster_tracker: a dictionary where the keys are the
                total number of clusters present in the agglomeration and
                the values are sets of the labels of the clusters present.
                This is useful for tracking progress of the aggolmeration.
                For example, self.clusters[self.cluster_tracker[10]] will
                return all the clusters present when the agglomeration reached
                10 remaining clusters.
            3. self.map_tracker: a dictionary where the keys are the number
                of total clusters left in the agglomeration and the values
                are numpy arrays indicating which SOM prototype belongs to
                which cluster. This is useful for making graphics displaying
                the progress of the agglomeration.
        """
        
        self.clusters = {}
        self.fringe = {}
        self.cluster_tracker = {}
        self.map_tracker = {}
        self.clusters_map = None
        self.current_clusters = None
        
        self.graph = graph
        self.nodes = sorted(graph.graph.nodes())
        
        print(self.nodes[:10])

        self._create_clusters()
        self.cluster_tracker[len(self.clusters)] = self.clusters
        self.current_clusters = set(self.clusters.keys())

        neighbors = self._find_neighbors()
        self._calculate_metrics(neighbors)

        while self.fringe.keys():
            
            # get the best pair off the fringe
            evalf = self._evaluate_fringe()
            for pair in evalf:
                if not self._check_connectivity(pair[0]):
                    del self.fringe[pair[0]]
                else:
                    self._merge(pair[0])
                    break
            
            # update the trackers
            lcc = len(self.current_clusters)
            self.cluster_tracker[lcc] = copy.copy(self.current_clusters)
            self.map_tracker[lcc] = np.copy(self.clusters_map)

    def _find_neighbors(self, specifics=None):
        
        """ Find all pairs of neighboring clusters in clusters_map """

        def _new(item, this_cluster, pairs):
            """ Helper """
            
            test1 = (item, this_cluster) not in pairs
            test2 = (this_cluster, item) not in pairs
            return test1 and test2

        def _make_iterlist():
            """ Helper """
            if specifics != None:
                tmp = [np.where(self.clusters_map == x, 1, 0) for x in specifics]
                where = np.nonzero(sum(tmp))
                iterlist = zip(where[0], where[1])
                
            if specifics == None:
                yvals = [x for x in range(self.som.grid_size[0])]
                xvals = [x for x in range(self.som.grid_size[0])]
                iterlist = itertools.product(yvals, xvals)
                
            return iterlist

        pairs, dist = set([]), 1
        for row, col in _make_iterlist():

            this_cluster = self.clusters_map[row, col]

            # get all the surrounding values
            rmin = max(0, row-dist)
            rmax = min(self.som.grid_size[0], row+dist+1)-1
            cmin = max(0, col-dist)
            cmax = min(self.som.grid_size[1], col+dist+1)-1

            found = set([self.clusters_map[rmin, col],
                         self.clusters_map[rmax, col],
                         self.clusters_map[row, cmin],
                         self.clusters_map[row, cmax]])
            
            found.discard(this_cluster)

            # add this pair of clusters if its not already in the set
            for item in found:
                if _new(item, this_cluster, pairs):
                    pairs.add((this_cluster, item))
    
        return pairs
        
    def _create_clusters(self):
        """ Set up initial clusters. This method
        creates self.clusters and self.clusters_map """
        
        def _iterator():
            """ Helper """
            x_range = [x for x in range(self.som.grid_size[0])]
            y_range = [y for y in range(self.som.grid_size[1])]
            return itertools.product(x_range, y_range)
        
        def _knn_fit():
            """ Helper """
            # now assign prototypes to the remaining clusters
            # by kNN as provided by sklearn. this forms the
            # initial clusters_map which the agglomerator will use
            # to find neighbors.
            idxy, idxx = np.indices(self.som.umatrix.shape, np.uint8)
            grid = np.c_[idxy.ravel(), idxx.ravel()]
            
            from sklearn.neighbors import KNeighborsClassifier
            k = self.clusters.keys()
            coords = [list(self.clusters[j].coordinates)[0] for j in k]
            labels = [self.clusters[j].label for j in k]
            knn = KNeighborsClassifier(weights='uniform', n_neighbors=1)
            fitted = knn.fit(coords, labels).predict(grid)
            
            fitted = fitted.reshape(self.som.umatrix.shape)
            return fitted
            
        
        # make sure we have a umatrix
        self.som.calculate_u_matrix(nearest=8)
        
        # instantiate a cluster at each site on the map
        j = 1

        for row, col in _iterator():
            cluster = self.cluster_obj(j, self.som.umatrix)
            cluster.add_coord((row, col))
            self.clusters[j] = cluster
            j += 1
            
        # for each sample in self.samples, find the winning
        # neuron and assign that sample to the corresponding
        # cluster. clusters start at 1; coordinates start at 0...
        for i, sample in enumerate(self.som.samples):
            winner = self.som._find_winner(sample)
            self.clusters[winner+1].add_tag(self.nodes[i])
                    
        # delete any clusters which do not have tags assigned
        to_delete = [key for key, cluster in self.clusters.items()
                     if not cluster.has_tags]
        for key in to_delete:
            del self.clusters[key]
        
        # generate the initial umetric for each cluster
        for obj in self.clusters.values():
            obj.calculate_u_total()

        self.clusters_map = _knn_fit()

        for row, col in _iterator():
            self.clusters[self.clusters_map[row, col]].add_coord((row, col))

    def _check_connectivity(self, pair):
        """ Check whether the subgraph composed of the union
        of the pair p is actually connected. DONT AGGLOMERATE
        UNCONNECTED SUBGRAPHS!"""
        
        tags0 = self.clusters[pair[0]].tags
        tags1 = self.clusters[pair[1]].tags
        tags = tags0.union(tags1)
        subgraph = self.graph.graph.subgraph(tags).to_undirected()
        return networkx.is_connected(subgraph)

    def _metric(self, cluster1, cluster2):
        """ Algorithm specific. Calculate the average
        umatrix value for the combined cluster."""
        
        return cluster1.u_total+cluster2.u_total

    def _calculate_metrics(self, pairs):
        """ For all uncalculated pairs in up not in the fringe,
        calculate their pairwise metric and add it to the fringe"""
        
        for item in pairs:
            if item not in self.fringe:
                cluster0 = self.clusters[item[0]]
                cluster1 = self.clusters[item[1]]
                self.fringe[item] = self._metric(cluster0, cluster1)
         
    def _evaluate_fringe(self):
        """ Return a fringe sorted by metric """
        fringe_list = [(k, v) for k, v in self.fringe.items()]
        fringe_list.sort(key=lambda z: z[1])
        return fringe_list
    
    def _merge(self, best):
        
        """ Merge the pair of clusters given in tuple best """
        
        def _old(pair):
            """ Helper for _merge """
            return cluster1 in pair or cluster2 in pair
        
        cluster1, cluster2 = best
        
        # create a new cluster from the two old clusters
        j = max(self.clusters.keys())+1
        newc = self.cluster_obj(j, self.som.umatrix)
        newc.merge_with((self.clusters[cluster1], self.clusters[cluster2]))
        self.clusters[j] = newc

        # update clusters_map
        self.clusters_map[self.clusters_map == cluster1] = j
        self.clusters_map[self.clusters_map == cluster2] = j
        
        # update current_clusters
        self.current_clusters.add(j)
        self.current_clusters.discard(cluster1)
        self.current_clusters.discard(cluster2)
        
        # remove old metrics
        to_remove = set([p for p in self.fringe.keys() if _old(p)])
        for key in to_remove:
            del self.fringe[key]
        
        # add new metrics
        if len(self.current_clusters) > 1:
            new_neighbors = self._find_neighbors(specifics=[j,])
            self._calculate_metrics(new_neighbors)

class SOM(object):
    
    """ Class for training a SOM and agglomerating a self-organizing map
    to classify feature vectors.
    
    
    Typical usage pattern:
    
    som = SOM((15,15)) # 15x15 neurons
    som.train(featureVectors,featureVectors.shape[0]*5)
    som.agglomerator.agglomerate(graph)
    som.agglometator.assign_tags_to_clusters()
    
    """
    
    def __init__(self, grid_size, agglomerator_obj=Agglomerator):
        
        """ Initialize the SOM; assign default values to various quantities.
        
        Requires:
            grid_size - a 2d tuple of integers which describes how many
                neurons to use along each dimension.
        """
        
        # right now, only support 2-dimensional grids
        assert len(grid_size) == 2
        self.grid_size = grid_size
        
        self.neighbors = {4:((-1, 0), (1, 0), (0, 1), (0, -1)),
                          8:((-1, -1), (-1, 0), (-1, 1), (0, -1), \
                            (0, 1), (1, -1), (1, 0), (1, 1))}
        
        # default learning parameters
        self.params = {'ratei':.4, 'ratef':0.05,
                       'sigmai':np.max(self.grid_size)/2,
                       'sigmaf':1, 'gammai':3, 'gammaf':0}
        
        self.a_o = agglomerator_obj
        
        # most of the init action takes place in this function,
        # which can also be called externally to clear everything
        self.reset()

    def reset(self):
        
        # stuff for umatrix
        self.umatrix = None

        # make the grid indices
        self.grid = np.mgrid[0:self.grid_size[0], 0:self.grid_size[1]]
        self.gidx = np.vstack((self.grid[1].ravel(), self.grid[0].ravel())).T

        self.n_nodes = self.grid_size[0]*self.grid_size[1]
        self.trained = False
        self.counts = np.zeros((self.n_nodes,), np.float32)
        
        # the agglomerator
        self.agglomerator = self.a_o(self)

        # stuff for frequency matrix
        self.frequency_matrix = None
        
        # stuff for unetwork
        self.unetwork = None
        self.distance_regularizer = 0.02
        self.umatrix_json = None
        
        # initial definitions for train
        self.weights = None
        self.is_sparse = None
        self.nsamples = None
        self.training_time = None
        self.ndim = None
        self.samples = None
        self.iterations = None
        
        # initial definitions for assign_tags_to_clusters
        self.labels = None
        self.assigned = None
        
    def train(self, samples, iterations, params=None):
        
        """ Train the SOM using samples as input feature vectors.
        
        One iteration of training consists of:
            1. Selecting a feature vector from samples
            2. Calculating the most-activated neuron
            3. Updating the weights of the SOM
        
        Required input:
            samples - an iterable of feature vectors, such as
                an adjacency matrix generated by networkX
            iterations - total number of iterations in the training
                process. On each iteration, a sample is selected
                at random and put through the trainer.
            
        Optional input:
            params - a dictionary of optional parameters used in
                training the SOM. Values can include:
                
                1. ratei (initial learning rate) default 0.4
                2. ratef (final learning rate) default 0.05
                3. sigmai (initial neighborhood size)  default max GridSize/2
                4. sigmaf (final neighborhood size) default 1
                5. gammai (initial similarity bias) default 3
                6. gammaf (final similarity bias) default 0
                
                
        """
        
        def _scale(start, end, k):
            """ Helper function to step the learning and neighborhood
            parameters""" 
            return start+k*(end-start)

        # when the samples come in we can make the weights
        self.samples = samples
        self.nsamples, self.ndim = samples.shape
        tmp = (self.n_nodes, self.ndim)
        self.weights = np.random.random(tmp).astype(np.float32)
        
        self.is_sparse = sparse.issparse(samples)
        
        # update the learning parameters versus defaults
        if params not in ('None', None):
            self.params.update(params)
        
        # this precomputes which samples will be used at each iteration
        idx = np.random.randint(0, len(samples), (iterations,))
        
        fiter = float(iterations)
        self.iterations = iterations
        
        for k in range(iterations):
        
            # get the sample for this iteration. if its sparse, convert it
            # to a 1 dimensional array
            sample = self.samples[idx[k], :]
            if self.is_sparse:
                sample = np.array(sample.todense())[0]

            # find the most representative neuron and update the weights
            par = self.params
            gamma = _scale(par['gammai'], par['gammaf'], k/fiter)
            rate = _scale(par['ratei'], par['ratef'], k/fiter)
            sigma = _scale(par['sigmai'], par['sigmaf'], k/fiter)
            winner = self._find_winner(sample, gamma=gamma)
            self._update_weights(sample, winner, rate, sigma)

        self.trained = True

    def calculate_u_matrix(self, reduction='sum', nearest=4, unetwork=False):
        """ Build the u-matrix from the weights. The value u_{i,j} at site
        (i,j)  is defined as reduction([d(w_{i,j}, w_{n,m})]) where d is
        the distance function, w are the weights, and {n,m} indicate the
        nearest neighbors to site (i,j).
        
        reduction is by default the mean but could also be some other
        function, like max().
        
        nearest is the number of nearest neighbors to use. if nn = 4, only use
        direct x or y neighbors. if nn = 8, also use diagonal neighbors."""
        
        slices = {-1:':-1', 0:':', 1:'1:'}
        
        nearest = self.neighbors[nearest]
        weights = self.weights.reshape(self.grid_size+(self.ndim,))
        distances = np.zeros(self.grid_size, np.float32)
        
        for neighbor in nearest:
            
            rld = np.roll(weights, neighbor[0], axis=0)
            rld = np.roll(rld, neighbor[1], axis=1)
            dist = np.sqrt(np.sum((weights-rld)**2, axis=-1))

            # because boundary conditions aren't circular, do
            # this crazy slicing using exec to update correctly.
            # can this be done without exec?
            s_exec = '[%s,%s]'%(slices[neighbor[0]], slices[neighbor[1]])
            if reduction == 'sum':
                tmp = (s_exec, s_exec)
                cmd = 'distances%s += dist%s'
            if reduction == 'max':
                tmp = (s_exec, s_exec, s_exec)
                cmd = 'distances%s = np.maximum(distances%s,dist%s)'
            exec(cmd%tmp)
            
        self.umatrix = distances
        
        tmp = (self.umatrix-self.umatrix.min())
        tmp = tmp/tmp.max()*255
        self.umatrix_json = tmp.astype(np.uint8).tolist()
        
    def calculate_frequency_matrix(self, vectors=None):
        
        """ Calculate the winning frequency of each neuron in the map.
        Obviously, the map must be trained before this can be calculated
        in a meaningful way."""
        
        if vectors == None:
            vectors = self.samples#[self.idx,:]
            
        from collections import Counter
        self.frequency_matrix = np.zeros(self.grid_size, np.int)
        if self.trained:
            # we can probably make this faster
            if self.is_sparse:
                function = find_winner_mp_sparse
            else:
                function = find_winner_mp
            work = [(v, self.weights, self.grid_size) for v in vectors]
            winners = Counter([function(w) for w in work])
            for coord, count in winners.items():
                self.frequency_matrix[coord] = count

    def report_tag_assignments(self, number_of_clusters):
        """ Report which tags have been assigned to which
        clusters when the agglomeration has generated
        n remaining clusters. """
        
        act = self.agglomerator.cluster_tracker
        
        if self.assigned:
            for cluster in act[number_of_clusters]:
                members = self.agglomerator.clusters[cluster].tag_members
                if self.labels != None:
                    tags = [self.labels[x] for x in members]
                    print("***\n%s\n"%tags)
                else:
                    print(members)
        else:
            pass

    def _find_winner(self, sample, gamma=0):
        """ Given a feature vector "sample", finds the most-activated
        neuron are returns its (1d) index.
        
        This function can take an optional parameter gamma, which
        attempts to evenly distribute winning neurons around the map
        by increasing the euclidean distance between a
        feature vector and the prototype neurons by an amount
        proportional to the number of feature vectors which have
        mapped onto that neuron already. (default 0) """

        # numexpr doesnt help here; the expression is too simple
        if self.is_sparse:
            sample = np.array(sample.todense()) # is this right?
        distances = np.sum((self.weights-sample)**2, axis=1)
        distances *= (1+gamma*np.sqrt(self.counts))
        
        return distances.argmin()

    def _update_weights(self, sample, winner, rate, sigma):
        """ Given a feature vector "sample", the winning neuron "winner",
        the learning rate "rate", and the (isotropic) gaussian neighborhood
        size "sigma", update the weights in the protype neurons"""
        
        # array manipulations in this function are somewhat complicated to
        # allow fast vector operations, achieving very high single-threaded
        # performance.
        
        dist = (self.gidx[:, 0]-self.gidx[winner, 0])**2+\
               (self.gidx[:, 1]-self.gidx[winner, 1])**2

        update = rate*np.exp(-1*dist/(2*sigma**2))
        update = update.reshape(update.shape+(1,))
        self.weights += update*(sample-self.weights)
        self.counts[winner] += 1
        
class MultiSOM(object):
    
    """ A generalization of the SOM class to allow for training
    and agglomerating several SOMs, then comparing the results
    of the agglomeration to find repeatable clusters.
    
    Introduces several new methods for analyzing the repeats.
    
    Doesn't inherit specifically from SOM, but instead
    creates SOMs as needed.
    """
    
    def __init__(self, grid_size, graph=None, processes=4):
        self.SOM = SOM(grid_size)
        self.grid_size = grid_size
        self.workers = Pool(processes)
        self.graph = graph
        
        # initial definitions for train
        self.som_params = None
        self.repeats = None
        self.samples = None
        
        # initial definitions for cluster_reproducibility
        self.reproduction_matrices = None
        self.reproduction_analysis = None
        
        # initial definitions for assign_prototypes_knn
        self.knn_clusters = None
        self.knn_borders = None
        self.umatrixlist = None
        self.umatrix = None
        
    def reset(self):
        self.SOM.reset()

    def train(self, samples, iterations, repeats, params=None, debug=False):
        """ Train several SOMs using samples as input feature vectors.
        
        One iteration of training a single SOM consists of:
            1. Selecting a feature vector from samples
            2. Calculating the most-activated neuron
            3. Updating the weights of the SOM
        
        This function uses multi-processing by default with 4 processes
        in order to reduce the amount of time required to train
        all the SOMs.
        
        Required input:
            samples - an iterable of feature vectors, such as
                an adjacency matrix generated by networkX
            iterations - total number of iterations in the training
                process. On each iteration, a sample is selected
                at random and put through the trainer.
            repeats - how many SOMs to train.    
                
        Optional input:
            params - a dictionary of optional parameters used in
                training the SOM. Values can include:
                
                1. ratei (initial learning rate) default 0.4
                2. ratef (final learning rate) default 0.05
                3. sigmai (initial neighborhood size)  default max GridSize/2
                4. sigmaf (final neighborhood size) default 1
                5. gammai (initial similarity bias) default 3
                6. gammaf (final similarity bias) default 0
                
            debug - If True, will run all the SOM training on a single
                CPU. If False, will do training using 4 processes.
                
        Returns:
        
            Nothing ----BUT---- generates a set of co-occurrence data
            assigned to self.repeats. This data is interpreted as follows:
            
            self.repeats[n,i,j] is the number of times tag_i and tag_j
            were agglomerated into the same cluster when the number of
            clusters remaining in the agglomeration was n. This number must
            always fall into the range [0,repeats]. To be more specific,
            tag_i and tag_j are the tags associated with feature vector i
            and feature vector j, respectively.  
        """

        if repeats%4 != 0:
            repeats += 4-repeats%4
        
        self.samples = samples
        self.iterations = iterations
        self.som_params = params
        
        if params == None:
            params = 'None'
        
        jobs = [(self.grid_size, int(repeats/4), self.samples,
                 iterations, params, self.graph) for dummy in range(4)]
        
        # the results array is a similarity figure
        if debug:
            self.repeats = sum([_train_mp(job) for job in jobs])
        else:
            print(jobs)
            self.repeats = sum(self.workers.map(_train_mp, jobs))
    
    def assign_prototypes_knn(self, samples=None, iterations=None,
                              params=None, analysis=None):
        
        """ Train a SOM and assign prototypes to the clusters found through
        cluster_reproducibility using KNN algorithm from sklearn"""

        def _borders(array):
            """ Given an array, generate the auxiliary array designating
            which cells are cluster borders """
            
            border = np.zeros(array.shape, np.int)

            border[:, -1] += 1 # right border
            border[:, 0] += 2 # left border
            border[0, :] += 4 # top border
            border[-1, :] += 8 # bottom border

            border[:, :-1] += 1*(array[:, :-1] != array[:, 1:])
            border[:, 1:] += 2*(array[:, :-1] != array[:, 1:])
            border[1:, :] += 4*(array[:-1, :] != array[1:, :])
            border[:-1, :] += 8*(array[:-1, :] != array[1:, :])

            return border
        
        def _make_classifier():
            """ Helper """
            # follow the recipe for using the knn classifier
            from sklearn.neighbors import KNeighborsClassifier
            idy, idx = np.indices(self.SOM.umatrix.shape, np.uint8)
            grid = np.c_[idy.ravel(), idx.ravel()]
            knn = KNeighborsClassifier(weights='uniform', n_neighbors=10)
            ncpu = len(self.workers._pool)
            
            return grid, knn, ncpu
            
        def _make_chunks():
            j = working['members'].items()
            entries = [(n, samples[i2]) for n, i in j for i2 in i]
            cuts = np.linspace(0, len(entries), ncpu+1).astype(int)
            chunks = [(entries[cuts[i]:cuts[i+1]], self.SOM.weights,
                       self.SOM.grid_size) for i in range(ncpu)]
            return chunks
        
        def _find_winners():
            tmp = self.workers.map(find_winners_mp, chunks)
            coords = [item[0] for sublist in tmp for item in sublist]
            labels = [item[1] for sublist in tmp for item in sublist]
            return coords, labels
            

        # first, train the som and build a umatrix
        if samples == None:
            samples = self.samples
        if iterations == None:
            iterations = self.iterations

        self.knn_clusters = []
        self.knn_borders = []
        self.SOM.train(samples, iterations, params)
        self.SOM.calculate_u_matrix()
        
        self.umatrix = (self.SOM.umatrix*255./self.SOM.umatrix.max())
        self.umatrix = self.umatrix.astype(np.uint8)
        self.umatrixlist = self.umatrix.tolist()

        grid, knn, ncpu = _make_classifier()        

        # for each entry in analysis, make a map of the winners
        if analysis == None:
            analysis = self.reproduction_analysis

        for working in analysis:
            # create a series of cluster labels and vector indices
            # to distribute to the workers. then slice them into
            # as many portions as there are workers. this minimizes
            # the transfers.
            chunks = _make_chunks()
            
            coords, labels = _find_winners()
            
            # classify all points in the SOM with the kNN algorithm.
            # code is adapated from the sklearn documentation.
            predicted = knn.fit(coords, labels).predict(grid)
            predicted = predicted.reshape(self.SOM.umatrix.shape)

            self.knn_clusters.append(predicted.tolist())
            self.knn_borders.append(_borders(predicted).tolist())

    def cluster_reproducibility(self, repeats=None, clusters=50):
        
        """ Given the tag co-occurence arrays generated by the train
        method, use the spectral clustering method in sklearn and the
        known (or desired) number of clusters to assign tags to
        specific clusters.
        
        Required input:
            None
            
        Optional input:
            repeats - a set of co-occurence arrays to cluster using
                spectral methods. If not supplied, this method
                defaults to self.repeats which is the data generated
                by the train() method.
                
            labels - the tags corresponding to the feature vectors.
                Labels must be correctly ordered, obviously.
                
        Returns:
        
            None ----BUT---- generates the following analysis in the
            self namespace.'
            
            1. self.reproduction_matrices: a reorganization of the
                repeats data into block diagonal form.
                
            2. self.reproduction_analysis: a list of dictionaries.
                Each dictionary has two keys: 'members' and 'sizes'.
                
                'members' lists the tag membership of each cluster
                in terms of the indices of the feature vectors represented
                by samples in train(),arranged by size.
                
                'sizes' gives the size of each
                cluster. The index of the self.reproduction_analysis
                list gives the number of clusters remainging from
                the agglomeration. For example,
                
                self.reproduction_analysis[10][4]['members'] lists the
                tag indices of the 5th largest cluster when there are
                11 clusters remaining from the agglomeration.
                
        
        """
    
        def _find(where, what):
            """ Helper """
            return np.where(where == what[0])[0].tolist()
    
        from sklearn.cluster import SpectralClustering
        from collections import Counter

        if repeats == None:
            repeats = self.repeats
        spectral = SpectralClustering(n_clusters=1, affinity="precomputed")

        cluster = 0
        
        shape = (clusters,)+repeats.shape[1:]
        self.reproduction_matrices = np.zeros(shape, np.uint8)
        self.reproduction_analysis = []

        for idx, repeat in enumerate(repeats[:clusters]):

            # run the spectral clustering on the current repeat array.
            # this is the rate limiting step, and already uses all
            # available cpu cores.
            spectral.set_params(n_clusters=idx+1)
            spectral.fit(repeat)
            labels = spectral.labels_

            # order the clusters by size. keys in members are strings
            # as required for json dumps
            count = Counter(spectral.labels_)
            by_size = [(k, v) for k, v in count.items()]
            by_size.sort(key=lambda x: -x[1])
            members = {str(t[0]+cluster):_find(labels, t) for t in by_size}
            order = np.hstack([members[str(t[0]+cluster)] for t in by_size])

            #rearrange
            rearr = repeat[order].transpose()[order]
            sizes = [[str(k), len(v)] for k, v in members.items()]
            sizes.sort(key=lambda x: -x[1])
            
            # m gives the counts for each pair of tags. 3d array.
            # shape: [nclusters-1,ntags,ntags]. members are the tag
            # indices; self.graph.graph.nodes()[members] gives members as words.
            # sizes are the number of tags in each cluster, sorted by size
            tmp = {'members':members, 'sizes':sizes}
            
            rescale = (rearr*255./rearr.max()).astype(np.uint8)
            self.reproduction_matrices[idx] = rescale
            self.reproduction_analysis.append(tmp)
            cluster += idx+1
