/* Voice-RAG Frontend Logic for Chat UI */

const startBtn = document.getElementById('startBtn');
const btnText = document.getElementById('btnText');
const micIcon = document.getElementById('micIcon');
const sendIcon = document.getElementById('sendIcon');
const textInput = document.getElementById('textInput');
const status = document.getElementById('status');
const connStatus = document.getElementById('connStatus');
const connIndicator = document.getElementById('connIndicator');
const transcript = document.getElementById('transcript');
const visualizer = document.getElementById('visualizer');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const resetBtn = document.getElementById('resetBtn');
const documentListDiv = document.getElementById('documentList'); // New element

// State
let ws;
let audioCtx;
let nextStartTime = 0;
let activeSources = [];
let isRecording = false;
let isBusy = false; // New busy state flag
let audioStream;

// Helper to manage busy state
function setBusy(busy) {
    isBusy = busy;
    startBtn.disabled = busy;
    textInput.disabled = busy;
    fileInput.disabled = busy;
    resetBtn.disabled = busy;
    if (busy) {
        status.innerText = "Processing...";
        startBtn.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
        status.innerText = "Ready to converse";
        startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

// Helper to load document list
async function loadDocumentList() {
    try {
        const res = await fetch('/api/knowledge/list');
        const data = await res.json();
        documentListDiv.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                const li = document.createElement('li');
                li.className = 'text-sm text-slate-400';
                li.innerText = `${doc.filename} (${doc.chunks} chunks)`;
                documentListDiv.appendChild(li);
            });
        } else {
            documentListDiv.innerHTML = '<li class="text-sm text-slate-500 italic">No documents uploaded.</li>';
        }
    } catch (err) {
        console.error("Failed to load document list:", err);
        documentListDiv.innerHTML = '<li class="text-sm text-rose-400 italic">Failed to load docs.</li>';
    }
}

// Initialize Visualizer
for(let i=0; i<32; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar w-1.5 bg-sky-500 rounded-full h-2 opacity-40';
    visualizer.appendChild(bar);
}

// Initial load of documents
loadDocumentList();

// Reset Logic
resetBtn.onclick = async () => {
    if (!confirm("Clear all uploaded documents?")) return;
    setBusy(true);
    try {
        const res = await fetch('/api/knowledge/reset', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            uploadStatus.className = "text-sm text-emerald-400";
            uploadStatus.innerText = "Knowledge Base Cleared";
            await loadDocumentList(); // Refresh list
        }
    } catch (err) {
        console.error("Reset failed", err);
        uploadStatus.className = "text-sm text-rose-400";
        uploadStatus.innerText = "Reset failed";
    } finally {
        setBusy(false);
    }
};

// File Ingestion
fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    setBusy(true);
    uploadStatus.className = "text-sm text-sky-400";
    uploadStatus.innerText = "Ingesting " + file.name + "...";
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/knowledge/ingest', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            uploadStatus.classList.replace('text-sky-400', 'text-emerald-400');
            uploadStatus.innerText = "Done! " + data.chunks + " chunks added.";
            await loadDocumentList(); // Refresh list
        } else {
            throw new Error();
        }
    } catch (err) {
        uploadStatus.classList.replace('text-sky-400', 'text-rose-400');
        uploadStatus.innerText = "Upload failed";
    } finally {
        setBusy(false);
    }
};

// Chat Helpers
function addMessage(text, role) {
    if (transcript.querySelector('.msg-bubble').innerText === "Hello! I'm Voice-RAG, your intelligent assistant. How can I help you today?") {
        transcript.innerHTML = ''; // Clear initial bot message
    }

    let msgDiv;
    if (role === 'bot' && transcript.lastElementChild?.dataset.role === 'bot-active') {
        msgDiv = transcript.lastElementChild.querySelector('.msg-bubble');
        msgDiv.innerText = text; // Update current bot message
    } else {
        msgDiv = document.createElement('div');
        msgDiv.className = role === 'bot' ? 'msg bot-msg' : 'msg user-msg';
        
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerText = text;
        msgDiv.appendChild(bubble);

        if (role === 'bot') msgDiv.dataset.role = 'bot-active';
        transcript.appendChild(msgDiv);
    }
    transcript.scrollTop = transcript.scrollHeight;
}

function stopPlayback() {
    activeSources.forEach(s => {
        try { s.stop(); } catch(e) {}
    });
    activeSources = [];
    nextStartTime = playbackCtx.currentTime;
}

// WebSocket Setup
startBtn.onclick = async () => {
    if (isBusy) return; // Prevent interaction if busy

    if (isRecording) { // Stop recording / end session
        if (audioStream) audioStream.getTracks().forEach(track => track.stop());
        if (ws) ws.close();
        location.reload(); // Simple refresh to reset state
        return;
    }

    // Start recording / start session
    setBusy(true); // Set busy during connection establishment
    try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            status.innerText = "Listening...";
            connStatus.innerText = "Connected";
            connStatus.classList.replace('text-slate-400', 'text-emerald-400');
            connIndicator.classList.replace('bg-red-500', 'bg-emerald-500');
            connIndicator.classList.add('shadow-emerald-500/50');
            btnText.innerText = "End Session";
            startBtn.classList.replace('bg-sky-500', 'bg-rose-500');
            startBtn.classList.replace('hover:bg-sky-400', 'hover:bg-rose-400');
            startBtn.classList.replace('shadow-sky-500/25', 'shadow-rose-500/25');
            isRecording = true;
            micIcon.classList.remove('hidden');
            sendIcon.classList.add('hidden');
            setBusy(false); // Clear busy state once connected
            startRecording(audioStream);
        };

        ws.onmessage = async (e) => {
            if (typeof e.data === 'string') {
                const data = JSON.parse(e.data);
                if (data.event?.textOutput) {
                    addMessage(data.event.textOutput.content, 'bot');
                    status.innerText = "Assistant is speaking...";
                } else if (data.event?.userTranscript) {
                    if (data.event.userTranscript.trim().length > 2) stopPlayback();
                    document.querySelectorAll('[data-role="bot-active"]').forEach(m => delete m.dataset.role);
                    addMessage(data.event.userTranscript, 'user');
                } else if (data.event?.statusUpdate) {
                    status.innerText = data.event.statusUpdate;
                } else if (data.event?.toolEvent) {
                    const toolName = data.event.toolEvent.name;
                    const toolStatus = data.event.toolEvent.status;
                    const statusMap = {
                        "search_documents": "Searching Knowledge Base...",
                        "web_search": "Searching the web...",
                        "calculator": "Performing calculation..."
                    };
                    status.innerText = statusMap[toolName] || `Agent using ${toolName}...`;
                }
            } else {
                playOutputAudio(e.data);
            }
        };

        ws.onclose = () => { location.reload(); };

    } catch (err) {
        alert("Microphone access denied or error: " + err.message);
        console.error(err);
        setBusy(false); // Clear busy state on error
    }
};

// Text Input Handling
textInput.onkeypress = async (e) => {
    if (isBusy) return; // Prevent interaction if busy
    if (e.key === 'Enter' && textInput.value.trim() !== '' && ws && ws.readyState === WebSocket.OPEN) {
        const message = textInput.value.trim();
        addMessage(message, 'user');
        ws.send(message);
        textInput.value = '';
        status.innerText = "Sending message...";
        stopPlayback();
        document.querySelectorAll('[data-role="bot-active"]').forEach(m => delete m.dataset.role);
    }
};

// Toggle Send/Mic Button based on input
textInput.oninput = () => {
    if (isBusy) return; // Prevent interaction if busy
    if (textInput.value.trim() !== '') {
        micIcon.classList.add('hidden');
        sendIcon.classList.remove('hidden');
    } else {
        micIcon.classList.remove('hidden');
        sendIcon.classList.add('hidden');
    }
};


function startRecording(stream) {
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);
    source.connect(processor);
    processor.connect(audioCtx.destination);

    const bars = document.querySelectorAll('.bar');
    processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const vol = inputData.reduce((a, b) => a + Math.abs(b), 0) / 512;
        
        bars.forEach((b, i) => {
            const h = 8 + (vol * 500 * Math.abs(Math.sin(i / 4 + Date.now() / 150)));
            b.style.height = Math.max(8, h) + 'px';
            b.style.opacity = Math.min(1, 0.3 + (vol * 5));
        });

        const output = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            output[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
        }
        if (ws.readyState === WebSocket.OPEN) ws.send(output.buffer);
    };
}

const playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
function playOutputAudio(arrayBuffer) {
    const int16 = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x7FFF;

    const buffer = playbackCtx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);
    const source = playbackCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(playbackCtx.destination);
    
    activeSources.push(source);
    source.onended = () => {
        const index = activeSources.indexOf(source);
        if (index > -1) activeSources.splice(index, 1);
    };

    const currentTime = playbackCtx.currentTime;
    if (nextStartTime < currentTime) nextStartTime = currentTime + 0.05;
    source.start(nextStartTime);
    nextStartTime += buffer.duration;
}
