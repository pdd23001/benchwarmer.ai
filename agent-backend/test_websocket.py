#!/usr/bin/env python3
"""
Manual test script to verify WebSocket connection works.
This simulates the frontend workflow by:
1. Creating a session
2. Connecting to the WebSocket
3. Verifying messages are received
"""

import asyncio
import websockets
import json
import requests
import sys
import os

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

async def test_websocket_connection():
    print("=" * 60)
    print("WebSocket Connection Test")
    print("=" * 60)
    
    # Step 1: Create a session by uploading a file
    print("\n[1] Creating session...")
    
    # Use the dummy.py file from the backend
    dummy_file_path = os.path.join(os.path.dirname(__file__), "dummy.py")
    
    if not os.path.exists(dummy_file_path):
        print(f"❌ Error: {dummy_file_path} not found")
        print("Creating a dummy file...")
        with open(dummy_file_path, "w") as f:
            f.write("# Dummy algorithm\ndef solve():\n    pass\n")
    
    with open(dummy_file_path, "rb") as f:
        files = {"file": ("dummy.py", f, "text/x-python")}
        response = requests.post(f"{BACKEND_URL}/api/session/start", files=files)
    
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.status_code}")
        print(response.text)
        return False
    
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"✅ Session created: {session_id}")
    print(f"   Detected class: {session_data.get('detected_class', 'unknown')}")
    
    # Step 2: Connect to WebSocket
    print(f"\n[2] Connecting to WebSocket...")
    ws_url = f"{WS_URL}/api/session/{session_id}/live"
    print(f"   URL: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Step 3: Wait for initial message
            print("\n[3] Waiting for messages...")
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"✅ Received message: {json.dumps(data, indent=2)}")
                
                if data.get("type") == "status" and data.get("status") == "connected":
                    print("✅ Connection confirmation received!")
                    return True
                else:
                    print(f"⚠️  Unexpected message type: {data.get('type')}")
                    return True  # Still connected, just different message
                    
            except asyncio.TimeoutError:
                print("⚠️  No message received within 5 seconds")
                print("   (This is OK if no benchmark is running)")
                return True  # Connection worked, just no messages
                
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ WebSocket connection failed with status code: {e.status_code}")
        return False
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nMake sure the backend is running on port 8000!")
    print("Run: cd agent-backend && python server.py\n")
    
    success = asyncio.run(test_websocket_connection())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: WebSocket connection works!")
    else:
        print("❌ TEST FAILED: WebSocket connection broken")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
