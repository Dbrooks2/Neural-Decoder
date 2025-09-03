"""
Use your phone's sensors as 'neural' input
Put phone on your head and tilt to control!
"""

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Store latest sensor data
sensor_data = {
    'alpha': 0,  # Tilt left/right
    'beta': 0,   # Tilt forward/back
    'gamma': 0,  # Rotation
}

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phone Brain Control</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial;
                text-align: center;
                background: #1a1a1a;
                color: white;
            }
            #status {
                font-size: 24px;
                margin: 20px;
            }
            #cursor {
                width: 300px;
                height: 300px;
                border: 2px solid #00ff00;
                margin: 20px auto;
                position: relative;
                background: #000;
            }
            #dot {
                width: 20px;
                height: 20px;
                background: #ff00ff;
                border-radius: 50%;
                position: absolute;
                transform: translate(-50%, -50%);
            }
        </style>
    </head>
    <body>
        <h1>Phone Brain Interface</h1>
        <p>Put phone on forehead, tilt to control!</p>
        <div id="status">Waiting for connection...</div>
        <div id="cursor">
            <div id="dot" style="left: 50%; top: 50%;"></div>
        </div>
        
        <script>
            let x = 150, y = 150;
            
            function handleOrientation(event) {
                // Send orientation to server
                fetch('/sensor_data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        alpha: event.alpha,
                        beta: event.beta,
                        gamma: event.gamma
                    })
                });
                
                // Update local cursor
                x = Math.max(0, Math.min(300, x + event.gamma * 0.5));
                y = Math.max(0, Math.min(300, y + event.beta * 0.5));
                
                document.getElementById('dot').style.left = x + 'px';
                document.getElementById('dot').style.top = y + 'px';
                
                document.getElementById('status').innerText = 
                    `Tilt: ${event.beta.toFixed(1)}° forward, ${event.gamma.toFixed(1)}° side`;
            }
            
            // Request permission for iOS
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission()
                    .then(response => {
                        if (response == 'granted') {
                            window.addEventListener('deviceorientation', handleOrientation);
                        }
                    });
            } else {
                window.addEventListener('deviceorientation', handleOrientation);
            }
        </script>
    </body>
    </html>
    '''

@app.route('/sensor_data', methods=['POST'])
def receive_sensor_data():
    global sensor_data
    sensor_data = request.json
    return jsonify({'status': 'ok'})

@app.route('/get_control')
def get_control():
    """Convert phone orientation to control commands"""
    # Tilt forward/back = forward/reverse
    # Tilt left/right = turn
    
    linear_vel = -sensor_data['beta'] / 90.0  # -1 to 1
    angular_vel = sensor_data['gamma'] / 90.0  # -1 to 1
    
    return jsonify({
        'linear': linear_vel,
        'angular': angular_vel,
        'raw': sensor_data
    })

if __name__ == '__main__':
    print("Starting Phone Brain Interface...")
    print("1. Connect your phone to the same WiFi")
    print("2. Open http://YOUR_COMPUTER_IP:5000 on your phone")
    print("3. Put phone on your forehead")
    print("4. Tilt head to control!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
