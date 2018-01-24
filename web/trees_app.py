from flask import Flask, render_template, request, jsonify
#import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('identifier.html')

if __name__ == '__main__':
    #threaded=True creates separate thread for each person who visits website
    #can set debug=True, but turn it OFF before deploying in production
    app.run(host='0.0.0.0', port='5002', threaded=True)
