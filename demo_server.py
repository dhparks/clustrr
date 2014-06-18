from __future__ import print_function
from __future__ import absolute_import
from flask import Flask, jsonify, request, send_from_directory, json, session, render_template
import os
import time
from datetime import timedelta
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
        for s_key in sessions.keys():
            if time0-sessions[s_key]['last'] > 3600*8:
                del sessions[s_key]
                
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
    """ Send the landing page to the browser """
    manage_session()
    print("serving session %s"%session['s_id'])
    return send_from_directory(".", "static/html/demo.html")

def error_page(kwargs):
    """ Send the error page to the browser """
    kwargs['img'] = random.choice(sadbabies)
    return render_template('error.html', **kwargs)

@app.route('/<project>/<cmd>', methods=['GET', 'POST'])
def dispatch_cmd(project, cmd):
    """ Dispatch a command from the browser to the backend """
    
    print(project, cmd)

    occ = 'executing a command'
    
    manage_session()
    
    # dispatch commands to the backend
    backend = sessions[session['s_id']]['backend']
    
    if backend == None:
        error = "expired session"
        kwargs = {'error':error, 'occasion':occ}
        return error_page(kwargs)

    from_backend = backend.cmds[cmd](request.args, request.json, request.form)
    return jsonify(**from_backend)

# for session management
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=60*8)

if __name__ == '__main__':
    sadbabies = glob.glob('static/error/sadbaby*.jpg')
    app.run(host="0.0.0.0", port=5004, debug=False)
    
