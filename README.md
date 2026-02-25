# Voice-RAG: AWS Nova Sonic Real-Time Voice Assistant

`Voice-RAG` is a high-performance, real-time voice assistant prototype powered by **AWS Bedrock Nova Sonic** (`amazon.nova-2-sonic-v1:0`).
 It features bidirectional streaming for low-latency, natural conversations, automated tool-calling, and smart interruption (barge-in) handling.

## 🚀 Key Features

-   **Nova Sonic Integration:** Leverages the latest bidirectional streaming capabilities of AWS Bedrock.
-   **Agentic Framework:** Built using the `strands-agents` SDK for robust session management and tool execution.
-   **Real-Time Audio:** Low-latency 16kHz input / 24kHz output with a scheduled jitter buffer in the browser for smooth playback.
-   **Integrated Tools:** Includes a native `calculator` and a `stop_conversation` verbal exit tool.
-   **Web UI:** Modern, responsive browser interface with a real-time audio visualizer.

## 🏗️ Architecture

-   **Backend:** FastAPI server (`main.py`) using `BidiAgent` to manage individual Nova Sonic sessions.
-   **Frontend:** Single-page application (`index.html`) using Web Audio API for capture and playback.
-   **Protocol:** WebSockets bridge raw PCM audio between the client and the agent.
-   **Authentication:** Supports SigV4 via standard AWS credentials or SSO profiles.

## 🛠️ Setup & Installation

### 1. Prerequisites
-   Python 3.12+ (managed with `uv` recommended).
-   AWS credentials with access to `amazon.nova-2-sonic-v1:0` in `us-east-1`, `eu-north-1`, or `ap-northeast-1`.

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```
*Note: If using SSO, ensure you have an active session via `aws sso login`.*

### 3. Install Dependencies
```bash
uv sync
```

## 🎮 Running the Application

1. Start the server:
   ```bash
   uv run main.py
   ```
2. Open [http://127.0.0.1:8001](http://127.0.0.1:8001) in your browser.
3. Click **Start Conversation** and speak!

## ⚙️ Advanced: Jitter Buffer
The web client implements a **Scheduled Playback Queue** to handle network jitter. This prevents audio "breaking" by calculating the precise `nextStartTime` for each audio chunk, ensuring gapless playback.

---
Built with ❤️ for real-time AI research.
