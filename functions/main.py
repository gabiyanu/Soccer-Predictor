"""
Soccer Predictor — Firebase Cloud Function entry point  (v2.0)
==============================================================
This file is intentionally minimal so that the Firebase CLI's 10-second
source-analysis window is never exhausted by slow imports.

All heavy logic (numpy, scipy, flask, Firebase Admin, Gemini, Sheets)
lives in predictor.py and is imported lazily on the first HTTP request.
"""

from firebase_functions import https_fn, options

_flask_app = None


def _get_flask_app():
    """Import predictor and build the Flask app on first request."""
    global _flask_app
    if _flask_app is None:
        from predictor import flask_app
        _flask_app = flask_app
    return _flask_app


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.MB_256,
    timeout_sec=60,
    invoker="public",
)
def api(req: https_fn.Request) -> https_fn.Response:
    """
    Firebase HTTPS Cloud Function.
    Delegates all /api/* routes to the Flask app in predictor.py.
    Firebase Hosting serves web/index.html and rewrites /api/** here.
    """
    app = _get_flask_app()
    with app.request_context(req.environ):
        return app.full_dispatch_request()
