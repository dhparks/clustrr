from __future__ import print_function
from __future__ import absolute_import

from flask import Flask, jsonify, request, redirect, send_from_directory, json, session, escape, render_template
from werkzeug import secure_filename
from flask.ext.compress import Compress
import os
import uuid
import shutil
from os.path import getctime

import re

import time
from datetime import timedelta
import numpy

import glob
import random

import interface.flickr_backend as ifb


app = Flask(__name__)

# concurrent user sessions are managed through a dictionary which holds
# the backends and the time of last activity
sessions = {}
def manage_session():
    # see if there is currently a session attached to the incoming request.
    # if not, assign one. the way to check for a session is to try to get a
    # key; an error indicates no session
        
    def _delete_old_sessions():
        # delete old sessions from the sessions dictionary.
        time0 = time.time()
        for sk in sessions.keys():
            if time0-sessions[sk]['last'] > 3600*8:
                del sessions[sk]
                
    def _make_new_session():
        
        # make a new uuid for the session
        s_id = str(time.time()).replace('.', '')[:12]
        t_id = int(s_id)
    
        # store these here in python; can't be serialized into the cookie!
        sessions[s_id] = {}
        sessions[s_id]['backend'] = ifb.Backend()
        sessions[s_id]['last'] = time.time()
    
        # store these in the cookie?
        session.permanant = True
        session['s_id'] = s_id
        
        return t_id

    try:
        s_id = session['s_id']
    except KeyError:
        ct = _make_new_session()
        _delete_old_sessions()

# the rest of the decorators are switchboard functions which take a request
# and send it to the correct backend
@app.route('/')
def serve_landing():
    # now send the landing page
    manage_session()
    print("serving session %s"%session['s_id'])
    return send_from_directory(".", "static/html/demo.html")

@app.route('/<page>.html', methods=['GET',])
def serve_static_page(page):
    """ Serve a static html page, for example tech.html """
    return send_from_directory(".", 'static/html/%s.html'%page)

@app.route('/images/<img>', methods=['GET',])
def serve_static_image(img):
    return send_from_directory(".", 'static/images/%s'%img)

@app.route('/<project>/<cmd>', methods=['GET', 'POST'])
def dispatch_cmd(project, cmd):
    print(cmd)

    occ = 'executing a command'
    
    manage_session()
    
    # dispatch commands to the backend
    backend = sessions[session['s_id']]['backend']
    
    if backend == None:
        error = "expired session"
        kwargs = {'error':"expired session",'occasion':occ}
        return error_page(kwargs)
    
    from_backend = backend.cmds[cmd](request.args, request.json, request.form)
    return jsonify(**from_backend)

# for session management
import os
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=60*8)

if __name__ == '__main__':
    Compress(app)
    app.run(host="0.0.0.0", port=5004, debug=False)
    
