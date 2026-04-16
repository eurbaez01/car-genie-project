import sys
import os
import traceback

# Add project root to path so app.py and src/ can be found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Change working directory to project root so relative data paths work
os.chdir(ROOT)

try:
    from app import app
except Exception as e:
    # Surface the real error in Vercel logs
    import flask
    _error_app = flask.Flask(__name__)

    @_error_app.route('/', defaults={'path': ''})
    @_error_app.route('/<path:path>')
    def catch_all(path):
        return flask.Response(
            f"<pre>Startup error:\n{traceback.format_exc()}</pre>",
            status=500,
            mimetype='text/html'
        )

    app = _error_app
