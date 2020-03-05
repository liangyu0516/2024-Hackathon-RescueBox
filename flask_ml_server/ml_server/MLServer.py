# -*- coding: utf-8 -*-
"""
    flask_ml.MLServer
    ~~~~~~~~~
    This module implements MLServer object.
    :copyright: 2020 Jagath Jai Kumar
    :license: MIT
"""

from flask import Flask, current_app, request, Response
from encoder_decoder.dtypes import InputTypes, OutputTypes
from encoder_decoder.default_config import encoders, decoders, extract_input, wrap_output

class MLServer(object):
    """The MLServer object is a wrapper class for the flask app object. It
    provides a decorator for turning a machine learning prediction function
    into a WebService on an applet.
    """



    def __init__(self, name):
        """Instantiates the MLServer object as a wrapper for the Flask app.
        Initializes '/' and '/get_available_models' as default rules.
        The landing page blocks machine learning functions from holding the
        default route. The '/get_available_models' returns the prediction
        functions that are being hosted b the server
        """
        self.app = Flask(name)

        @self.app.route("/get_available_models",methods=['GET'])
        def get_models():
            """Returns a list of models as a JSON object
            Format: {"result":['function1','function2',...]}
            """

            # routes that are held for the server
            prebuilt_routes=["/get_available_models","/static/<path:filename>"]
            routes = []
            for rule in self.app.url_map.iter_rules():
                if not str(rule) in prebuilt_routes:
                    routes.append('%s' % str(rule)[1:])

            # return routes as a pickled json object
            response = return_response({},routes)
            return Response(response=response)


    def route(self, rule, input_type:InputTypes, output_type:OutputTypes=OutputTypes.STRING):
        def build_route(ml_function):
            @self.app.route(rule,endpoint=ml_function.__name__,methods=['POST'])
            def prep_ML():
                input_data = decoders[input_type](extract_input[input_type](request))
                result = ml_function(input_data)
                output = {}
                wrap_output[output_type](encoders[output_type](result), output) # TODO Any problem with inplace append to dict?
                response = create_response(output)
                response = Response(response=response, status=200, mimetype="application/json")
                return response
            return prep_ML
        return build_route

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        """Runs the application on a local development server.

        Do not use ``run()`` in a production setting. It is not intended to
        meet security and performance requirements for a production server.
        Instead, see :ref:`deployment` for WSGI server recommendations.


        If the :attr:`debug` flag is set the server will automatically reload
        for code changes and show a debugger in case an exception happened.


        If you want to run the application in debug mode, but disable the
        code execution on the interactive debugger, you can pass
        ``use_evalex=False`` as parameter.  This will keep the debugger's
        traceback screen active, but disable code execution.


        :param host: the hostname to listen on. Set this to ``'0.0.0.0'`` to
            have the server available externally as well. Defaults to
            ``'127.0.0.1'`` or the host in the ``SERVER_NAME`` config variable
            if present.
        :param port: the port of the webserver. Defaults to ``5000`` or the
            port defined in the ``SERVER_NAME`` config variable if present.
        :param debug: if given, enable or disable debug mode. See
            :attr:`debug`.
        :param load_dotenv: Load the nearest :file:`.env` and :file:`.flaskenv`
            files to set environment variables. Will also change the working
            directory to the directory containing the first file found.
        :param options: the options to be forwarded to the underlying Werkzeug
            server. See :func:`werkzeug.serving.run_simple` for more
            information.
        """
        self.app.run(host, port, debug, load_dotenv, **options)