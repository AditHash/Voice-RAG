/* Voice-RAG Frontend Logic */

const startBtn = document.getElementById('startBtn');
const btnText = document.getElementById('btnText');
const status = document.getElementById('status');
const connStatus = document.getElementById('connStatus');
const connIndicator = document.getElementById('connIndicator');
const transcript = document.getElementById('transcript');
const visualizer = document.getElementById('visualizer');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const resetBtn = document.getElementById('resetBtn');

// State
let ws;
let audioCtx;
let nextStartTime = 0;
let activeSources = [];

// Initialize Visualizer
for(let i=0; i<32; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar w-1.5 bg-sky-500 rounded-full h-2 opacity-40';
    visualizer.appendChild(bar);
}

// Reset Logic
resetBtn.onclick = async () => {
    if (!confirm("Clear all uploaded documents?")) return;
    try {
        const res = await fetch('/api/knowledge/reset', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            uploadStatus.className = "mt-4 text-[11px] font-medium text-emerald-400 uppercase tracking-wider";
            uploadStatus.innerText = "Knowledge Base Cleared";
        }
    } catch (err) {
        console.error("Reset failed", err);
    }
};

// File Ingestion
fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    uploadStatus.className = "mt-4 text-[11px] font-medium text-sky-400 uppercase tracking-wider";
    uploadStatus.innerText = "Ingesting " + file.name + "...";
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/knowledge/ingest', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            uploadStatus.classList.replace('text-sky-400', 'text-emerald-400');
            uploadStatus.innerText = "Done! " + data.chunks + " chunks added.";
        } else {
            throw new Error();
        }
    } catch (err) {
        uploadStatus.classList.replace('text-sky-400', 'text-rose-400');
        uploadStatus.innerText = "Upload failed";
    }
};

// Chat Helpers
function addMessage(text, role) {
    if (transcript.querySelector('.italic')) transcript.innerHTML = '';

    let msgDiv;
    if (role === 'bot' && transcript.lastElementChild?.dataset.role === 'bot-active') {
        msgDiv = transcript.lastElementChild;
    } else {
        msgDiv = document.createElement('div');
        msgDiv.className = role === 'bot' 
            ? 'msg self-start bg-slate-800 text-sky-100 px-5 py-3 rounded-2xl rounded-bl-none max-w-[85%] border border-slate-700/50 shadow-sm'
            : 'msg self-end bg-sky-500 text-slate-950 font-semibold px-5 py-3 rounded-2xl rounded-br-none max-w-[85%] shadow-lg shadow-sky-500/10';
        
        if (role === 'bot') msgDiv.dataset.role = 'bot-active';
        transcript.appendChild(msgDiv);
    }
    msgDiv.innerText = text;
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
    if (ws) { location.reload(); return; }

    try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
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
            startRecording(stream);
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
                }
            } else {
                playOutputAudio(e.data);
            }
        };

        ws.onclose = () => { location.reload(); };

    } catch (err) {
        alert("Microphone access denied or error: " + err.message);
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
