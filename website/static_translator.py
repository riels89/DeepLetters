from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def capture_video():
    return send_from_directory('','capture_video.html')
    #return "test"
    
if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)