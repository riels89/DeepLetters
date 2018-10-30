from flask import Flask, send_from_directory, request, render_template
import json
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import run_net
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def capture_video():
    if request.method == 'POST':
        json_data = request.get_json(force=True)
        run_net.analyze(json_data['image'])

    else:
        if request.args.get('prediction') is not None:
            return render_template('capture_video.html', value=request.args.get('prediction'))
        else:
             return render_template('capture_video.html')
    #return "test"

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)