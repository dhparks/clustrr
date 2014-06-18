from __future__ import print_function
from __future__ import absolute_import

import time
import datetime
import re
import json # faster json libraries available, but getting the data is much slower than parsing it
#from multiprocessing import Pool

class FlickrError(Exception):
    """ Rename the generic exception class so we know its coming from here!"""
    pass

class QueryList(object):
    
    """ Keeps track of when flickr is queried to help with throttling """
    
    def __init__(self):
        self.record = []
        self.limit = 3600 # set on Flickr's end
        
    def add(self):
        """ Add a time to the record """
        self.record.append(time.time())
        
    def count(self, window=3600):
        
        """ Counts how many queries have passed through this interface
        in the previous number of specified seconds. Number of seconds
        is given by the "window" argument. Default 3600 = 1 hour. """
        
        if window < 0:
            return len(self.record)
        
        now, count = time.time(), 0
        for query in self.record:
            if now-query < window:
                count += 1
        return count
    
    def purge(self, window=3600):
        """ Get rid of old records """
        now, count = time.time(), 0
        while now - self.record[count] > window:
            count += 1
        self.record = self.record[count:]
        
    def check(self):
        """ Check if OK to query Flickr """
        # returns boolean True or False if it is safe to query flickr.
        # this exists so as not to overload the api key and get it
        # turned off.
        return self.count() < self.limit

class FlickrQ(object):
    
    """ Class for querying flickr and cracking the json
    response into more manageable datastructures. In general,
    we want back lists of dictionaries (each photo giving
    a dictionary of information)."""
    
    standard_extras = ('description', 'license', 'date_upload', 'date_taken', 'owner_name', \
        'icon_server', 'original_format', 'last_update', 'geo', 'tags', 'machine_tags', \
        'o_dims', 'views', 'media', 'path_alias', 'url_sq', 'url_t', 'url_s', \
        'url_q', 'url_m', 'url_n', 'url_z', 'url_c', 'url_l', 'url_o') \
    
    # allowed extras is taken from flickr api documentation
    allowed_extras = {
        'people_getPublicPhotos':standard_extras,
        'photos_search':standard_extras,
        'interestingness_getList':standard_extras
    }

    def __init__(self, session=None, key=None):
        if session != None:
            self.session = session
        if key != None:
            self.key = key
        self.query_times = QueryList()
        #self.workers = Pool(4)
    
    def _query(self, **kwargs):
        """ Helper: query flickr """
        # execute the method defined in self.query with
        # the parameters passed in kwargs. increment
        # the counter
        self.query_times.add()
        return self.query(**kwargs)
    
    def _parse(self, response, fields, extras=None, overrides=None, store=None, returns=False):

        """ Helper: parse the JSON returned by Flickr """

        if store == None:
            store = self.parsed
        if extras == None:
            extras = {}
        if overrides == None:
            overrides = {}
        if response[:14] == 'jsonFlickrApi(':
            response = response[14:-1]

        decoded = json.loads(response.decode())['photos']
        pages, total, photos = decoded['pages'], decoded['total'], decoded['photo']
        
        tmp = [_parse_flickr((photo, fields, extras, overrides)) for photo in photos]
        store += tmp
        
        #store += self.workers.map(_parse_flickr, [(p,fields,extras) for p in photos]) # multiprocessing version, runs about 30% faster

        if returns:
            return {'pages':pages, 'total':total, 'n':len(photos)}
        else:
            pass

    def user_from_url(self, url, session=None):
        """ Lookup a user on flickr by the URL of their photopage
        or profile. For example, if url is "flickr.com/photos/parskdh"
        then the returned uid should be 8073513@N03. """
        
        if session == None and self.session == None:
            return None
        if session != None:
            self.session = session

        # validate the url
        if 'flickr.com/photos/' not in url:
            return None # actually we should raise an error

        self.query = self.session.urls_lookupUser
        tmp = self._query(url=url, format='json')[14:-1]
        print(tmp)
        decoded = json.loads(tmp.decode())
        if decoded['stat'] == 'fail':
            raise FlickrError({'function':'user_from_url', 'message':decoded['message']})
        elif decoded['stat'] == 'ok':
            nsid = decoded['user']['id']
            uname = decoded['user']['username']['_content']
            return nsid, uname
        else:
            raise FlickrError({'function':'user_from_url', 'message':'unknown error'})

    def user_photos(self, nsid=None, session=None, fields=None, extras='', stop_date=None):

        """ Download a user's public photos. 
        
        returns the following datastructure by default:
        
        [{'id':$id, 'date_taken':$timestamp, 'tags':['list','of','tags'], otherfields:$something...}
         {...},
          ...
        ]
        
        more things will be returned if the 'extras' field is specified.
        Extras must, of course, conform to the list of options provided
        by Flickr.
        
        """
        
        # some kwargs are required for a query
        if nsid == None:
            return None
        if session == None and self.session == None:
            return None
        if session != None:
            self.session = session
        
        # check types
        assert '@' in nsid
        assert isinstance(extras, str)
        
        # build the extras field
        extras = self._check_extras('people_getPublicPhotos', extras)
        fields += extras.split(',')
        fields = list(set(fields))
        
        # run the query to completion. this basically duplicates the flickrapi
        # walk functionality, but i want to record the number of queries for
        # future use
        page = 1
        pages = 1e6
        per_page = 500
        self.query = self.session.people_getPublicPhotos
        self.parsed = []

        def _search():
            """ Helper: query flickr """
            return self._query(user_id=nsid, page=page, extras=extras, per_page=per_page, format='json')

        while page <= pages:
            print(page)
            returned = self._parse(_search(), fields, returns=True)
            pages = returned['pages']
            page += 1
            
            # we can also stop the queries if we're now
            # just repeating ourselves
            if stop_date != None:
                if stop_date > min([p['date_taken'] for p in self.parsed]):
                    print("hit stop date")
                    break

        return self.parsed

    def check_date(self, nsid=None, session=None, stop_date=None):

        """ Get the first photo in a user's stream. If date_upload is larger
        than stop_date, return True. This signals that we need to download more
        photos. This is just a modified user_photos function.
        """

        # some kwargs are required for a query
        if nsid == None:
            return None
        if session == None and self.session == None:
            return None
        if session != None:
            self.session = session
        
        # check types
        assert '@' in nsid
        
        # build the extras field
        extras = 'date_upload'
        fields = ['date_upload',]
        
        # run the query to completion. this basically duplicates the flickrapi
        # walk functionality, but i want to record the number of queries for
        # future use
        page = 1
        self.query = self.session.people_getPublicPhotos
        self.parsed = []

        def _search():
            """ Helper: query flickr """
            return self._query(user_id=nsid, page=page, per_page=1, extras=extras, format='json')

        self._parse(_search(), fields, returns=False)
        
        if self.parsed == []:
            print("error parsing")
            raise FlickrError({'function':'check_date', 'message':'This user has no public photos'})
        
        return self.parsed[0]['date_upload'] > stop_date
    
    def _prep_times(self, mindate, maxdate):
        
        """ Helper: for time-resolved queries, get the right datetime
        objects from mindate and maxdate """
        
        # do all the datetime stuff. return a list of tuples
        # each of which is a (start, end) pair
        try:
            min_stamp = stamp(mindate)
        except ValueError:
            min_stamp = stamp(mindate+' 00:00:00')
            
        try:
            max_stamp = stamp(maxdate)
        except ValueError:
            max_stamp = stamp(maxdate+' 00:00:00')
            
        dmin = datetime.date.fromtimestamp(float(min_stamp))
        dmax = datetime.date.fromtimestamp(float(max_stamp))
        delta = datetime.timedelta(days=1)
        total = (dmax-dmin).days

        # build the tuples
        days = [(dmin+delta*day, dmin+delta*(day+1)) for day in range(total)]
        return days
    
    def location_photos(self, session=None, key=None, start=None, end=None, where=None, lat=None, lon=None, radius=10, extras='', limit=1000, fields=None):
        
        """ A generator which walks over the flickr time/geo search.
        
        # search for photos by date and location
        # date is not required, but if used it must be through kwargs start and end.
        # both start and end must be strings of format YYYY-MM-DD
        # location is required
        # location method 1: a string where = string(something) which can be geocoded eg "Berkeley, CA, USA"
        # location method 2: a pair of lat, lon points eg lat = -122.1, lon = 45.02
        # if using a place name like "Berkeley CA USA", it will be turned into a lat, lon pair
        # if using lat, lon, a floating-point radius = float(something) can also be supplied
        
        """

        def _check_types(session, key, start, end, where, lat, lon, radius, extras, limit, fields):
            """ Helper: check types of incoming arguments """
            
            types = {'string':str, 'float': float, 'int': int}
            
            def cast(var, var_type):
                """ Helper: cast to desired data type""" 
                try:
                    return types[var_type](var)
                except:
                    raise ValueError("cant cast %s to %s"%(var, var_type))
            
            if session == None:
                if self.session != None:
                    session = self.session
                    
            if session != None:
                self.session = session
                    
            if key == None:
                if self.key != None:
                    key = self.key
                    
            if key != None:
                self.key = key
                    
            if start != None:
                start = cast(start, 'string')
            if end != None:
                end = cast(end, 'string')
            if where != None:
                where = cast(where, 'string')
            if lat != None:
                lat = cast(lat, 'float')
            if lon != None:
                lon = cast(lon, 'float')
            if radius != None:
                radius = cast(radius, 'float') 

            extras = cast(extras, 'string')
            limit = cast(limit, 'int')
            
            if fields == None:
                fields = []
            
            assert isinstance(fields, (list, str))
            if isinstance(fields, str):
                fields = fields.split(',')
            for field in fields:
                assert isinstance(field, str)
            
            assert where != None or (lat != None and lon != None)
                
            return session, key, start, end, where, lat, lon, radius, extras, limit, fields
        
        def _get_place(place):
            """ Helper: find a place using Flickr's geo lookup """
            places = [p for p in self.session.places_find(api_key=self.key, query=place).iter('place')]
            return places[0].get('latitude'), places[0].get('longitude')

        def _prep_extras(extras):
            """ Helper: make sure all the requested database fields are correct """
            extras = self._check_extras('photos_search', extras)
            return ','.join(list(set(extras.split(','))))

        def _search(start, end):
            """ Helper: query flickr """
            # we can get by without passing any arguments to this
            # function because everything is defined in the
            # scope above
            time.sleep(2)
            return self._query(
                lat=lat,
                lon=lon,
                page=page,
                extras=extras,
                format='json',
                api_key=self.key,
                per_page=per_page,
                safe_search=2,
                min_taken_date=start,
                max_taken_date=end)

        # check types and required kwargs
        session, key, start, end, where, lat, lon, radius, extras, limit, fields = _check_types(session, key, start, end, where, lat, lon, radius, extras, limit, fields)
        if session == None or key == None:
            yield None
        
        # do prep work for the query
        self.parsed = []
        extras = _prep_extras(extras)
        days = self._prep_times(start, end)
        fields += extras.split(',')
        fields = list(set(fields))
        self.query = self.session.photos_search
        if where != None:
            lat, lon = _get_place(where)
        
        # for each day, try to download up to limit photos
        for (start, end) in days:
            page, max_page, per_page = 0, 1e6, 250
            self.todays_photos = []
            while page < max_page:
                page += 1
                returned = self._parse(_search(start, end), fields, extras={'geo':(lat, lon)}, returns=True, store=self.todays_photos)
                max_page = min([returned['pages'], limit/per_page])
                print("%s: %.4d photos; page %s of %s; got %s"%(start, int(returned['total']), page, returned['pages'], returned['n']))
            yield self.todays_photos

    def explore_photos(self, session=None, start=None, end=None, fields=None, extras='', have_days=None):
        """ Get photos from explore between start and end """
        
        import urllib2
        
        # some kwargs are required for a query
        if session == None and self.session == None:
            yield None
        if session != None:
            self.session = session
        
        # check types
        assert isinstance(extras, str)
        assert not isinstance(start, type(None))
        assert not isinstance(end, type(None))
        
        # build the extras field
        extras = self._check_extras('people_getPublicPhotos', extras)
        fields += extras.split(',')
        fields = list(set(fields))
        
        # have_days should be a set for fast lookups
        if have_days == None:
            have_days = set([])
        else:
            have_days = set(have_days)
        
        # run the query to completion. this basically duplicates the flickrapi
        # walk functionality, but i want to record the number of queries for
        # future use
        page = 1
        per_page = 200
        days = self._prep_times(start, end)
        self.query = self.session.interestingness_getList
        self.parsed = []

        def _search():
            """ Helper: query flickr """
            print("explore for date = %s"%start)
            try:
                print("querying")
                response = self._query(page=page, date=start, extras=extras, per_page=per_page, format='json')
                if response == None:
                    print("got no response, try again")
                    time.sleep(2)
                    _search()
                else:
                    return response
            except urllib2.HTTPError:
                print("HTTPError; sleep then retry")
                time.sleep(2)
                _search()
            except Exception as e:
                print("unknown exception")
                print(type(e))
                time.sleep(2)
                _search()

        # iterate over all the days. if we have results for that day
        # already, skip it
        for (start, end) in days:
            self.todays_photos = []
            if round_stamp(stamp(start)) not in have_days:
                self._parse(_search(), fields,
                            returns=True,
                            store=self.todays_photos,
                            overrides={'date_upload':round_stamp(stamp(start))})
            yield self.todays_photos
        
    def _check_extras(self, query_type, extras):
        """ Helper: validate extras """
        allowed = self.allowed_extras[query_type]
        return ','.join([x for x in extras.split(',') if x in allowed])
    
def _parse_flickr(args):
    """ Parse the flickr response into a correctly-formed tuple
    for tag_cloud to add to the database """

    # separated into a different function for multiprocessing
    
    def _tokenize(field, text):
        """ Helper: tokensize the text for field """
        # really basic cleaning
        # 0. make lower case
        # 1. get rid of apostrophes
        # 2. get rid of html tags (links usually) in description
        text = re.sub("\\n", ' ', text)
        text = text.replace("'", "")
        if field == 'tags':
            text = text.replace(':', '_').replace('=', '_').replace('{','').replace('}','')
        if field == 'description':
            text = re.sub('<[^<]+?>', ' ', text)
        return text

    def _get(field, photo):
        """ Helper :
         not all fields come back the way we want, as correctly-formatted
         dictionary keys or attributes. this function handles the
         transformation from json to strings for the database. """

        fmts = {'thumbnail':'http://farm%(farm)s.staticflickr.com/%(server)s/%(id)s_%(secret)s_q.jpg',
                'geo':'(%(latitude)s,%(longitude)s)',
                'o_dims':'(%(o_width)s,%(o_height)s)',
                'url':'http://www.flickr.com/%(owner)s/%(id)s'}

        # these are the text fields to tokenize
        if field in ('description', 'tags', 'title'):
            try:
                txt = photo[field]
                if field == 'description':
                    txt = txt['_content']
            except:
                return ''
            try:
                return _tokenize(field, txt)
            except:
                return ''
            
        # these fields are built from other keys
        elif field in ('thumbnail', 'geo', 'o_dims', 'url'):
            try:
                return fmts[field]%photo
            except:
                return ''
            
        # date_taken requires a format change: sql -> timestamp
        # both date_taken and date_upload need to return integers
        elif field == 'date_taken':
            try:
                return int(stamp(photo['datetaken']))
            except:
                return 0
            
        elif field == 'date_upload':
            try:
                return int(photo['dateupload'])
            except:
                return 0

        elif field == 'source':
            return 'flickr'
        
        else:
            try:
                return photo[field]
            except:
                return ''
    
    def main(photo, fields, extras, overrides):
        """ Helper: dispatch arguments to _get """
        parsed = {field:_get(field, photo) for field in fields}
        for extra_key, extra_val in extras.items():
            if extra_key in fields and parsed[extra_key] == '':
                parsed[extra_key] = extra_val
        for override_key, override_val in overrides.items():
            if override_key in fields:
                parsed[override_key] = override_val
        return parsed

    return main(*args)

def stamp(date):
    """ turn a YYYY-MM-DD HH:MM:SS into a unix datestamp """
    if isinstance(date, datetime.date):
        date = str(date)
    try:
        return time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timetuple())
    except ValueError:
        return time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d').timetuple())

def round_stamp(timestamp):
    """ reduce a timestamp in precision to the nearest day """
    fmt = '%Y-%m-%d'
    calendar = datetime.datetime.fromtimestamp(int(timestamp)).strftime(fmt)
    return int(time.mktime(datetime.datetime.strptime(calendar, fmt).timetuple()))
    


