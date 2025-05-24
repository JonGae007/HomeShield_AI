from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def generate_frames():
    # Stream-URL der Netzwerk-Kamera
    stream_url = "http://192.168.186.157:8080/?action=stream"
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Fehler: Konnte keine Verbindung zum Stream herstellen")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Konvertiere das Frame in JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Sende das Frame als Teil des Multipart-Response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Server startet auf http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)