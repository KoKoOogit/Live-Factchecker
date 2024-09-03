import asyncio
import websockets
import numpy as np
from AudioProcessor import AudioProcessor
# Function to handle incoming WebSocket connections
ap = AudioProcessor()

chunks = np.array([], dtype=np.int16)


async def receive_audio(websocket, path):
    global chunks
    print("Client connected")
    try:
        while True:
            # Receive the raw audio data sent from the client
            audio_data = await websocket.recv()

            # Convert the received data (bytes) to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Concatenate the new audio data with existing chunks
            chunks = np.concatenate((chunks, audio_array))

            # Calculate the duration of the audio data in seconds
            duration = len(chunks) / 44100

            # Check if the duration is approximately 10 seconds
            if duration >=10:
                ap.start_process_from_nparr(chunks)
                print("Processed 10-second audio data:", chunks)
                chunks = np.array([], dtype=np.int16)  # Reset chunks

            # Print the received audio data for debugging
            # print("Received audio data:", audio_array)
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected", e)

# Start WebSocket server


async def start_server():
    print("Starting server...")
    server = await websockets.serve(receive_audio, "localhost", 6789)
    await server.wait_closed()

# Run the server
asyncio.run(start_server())
