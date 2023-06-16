import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

HOST = '127.0.0.1:5005'
URI = f'ws://{HOST}/api/v1/stream'

async def connect_to_server(uri):
    return await websockets.connect(uri, ping_interval=None)

async def send_to_server(websocket, message):
    await websocket.send(message)

    while True:
        incoming_data = await websocket.recv()
        incoming_data = json.loads(incoming_data)

        match incoming_data['event']:
            case 'text_stream':
                yield incoming_data['text']
            case 'stream_end':
                yield None

async def print_response_stream():
    conversation = []
    websocket = await connect_to_server(URI)
    # prePrompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    while True:
        user_input = input("\nEnter your question: ")
        previous_response=''
        # append the user input to the conversation
        conversation.append({'role': 'Instruction', 'message': user_input})
        prompt = "\n".join([f"### {turn['role']}: {turn['message']}" for turn in conversation])
        prompt += "\n### Response:"
        
        async for response in send_to_server(websocket, prompt):
            if response is not None:
                print(response, end='')
                sys.stdout.flush()
                previous_response+=response
            else:
                assistant_response = previous_response.split("Response:")[-1].strip()
                conversation.append({'role': 'Response', 'message': assistant_response})
                previous_response=''
                break

    await websocket.close()

if __name__ == '__main__':
    asyncio.run(print_response_stream())
