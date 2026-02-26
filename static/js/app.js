/* Voice-RAG Frontend Logic */

const startBtn = document.getElementById('startBtn');
const btnText = document.getElementById('btnText');
const status = document.getElementById('status');
const connStatus = document.getElementById('connStatus');
const connIndicator = document.getElementById('connIndicator');
const transcript = document.getElementById('transcript');
const visualizer = document.getElementById('visualizer');
const textInput = document.getElementById('textInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const chatIdDisplay = document.getElementById('chatIdDisplay');
const mediaInput = document.getElementById('mediaInput');
const mediaStatus = document.getElementById('mediaStatus');
const mediaList = document.getElementById('mediaList');
const clearMediaBtn = document.getElementById('clearMediaBtn');
const resetBtn = document.getElementById('resetBtn');
const localeSelect = document.getElementById('localeSelect');
const polyglotToggle = document.getElementById('polyglotToggle');
const voiceGenderSelect = document.getElementById('voiceGenderSelect');
const voiceSelect = document.getElementById('voiceSelect');
const assistantLangSelect = document.getElementById('assistantLangSelect');
const codeSwitchToggle = document.getElementById('codeSwitchToggle');
const endpointingSelect = document.getElementById('endpointingSelect');
const inputRateSelect = document.getElementById('inputRateSelect');
const outputRateSelect = document.getElementById('outputRateSelect');
const temperatureRange = document.getElementById('temperatureRange');
const topPRange = document.getElementById('topPRange');
const maxTokensInput = document.getElementById('maxTokensInput');
const themeToggle = document.getElementById('themeToggle');

const THEME_STORAGE_KEY = "voice_rag_theme";

function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === "light") root.classList.remove("dark");
    else root.classList.add("dark");
}

function initTheme() {
    try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (stored === "light" || stored === "dark") applyTheme(stored);
    } catch (_) {}

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const isDark = document.documentElement.classList.contains("dark");
            const next = isDark ? "light" : "dark";
            applyTheme(next);
            try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (_) {}
        });
    }
}

function setUploadStatus(message, tone = "info") {
    if (!uploadStatus) return;
    const base = "mt-4 text-[11px] font-medium uppercase tracking-wider";
    const toneClass =
        tone === "success" ? "text-emerald-600 dark:text-emerald-400" :
        tone === "error" ? "text-red-600 dark:text-red-400" :
        tone === "warn" ? "text-amber-600 dark:text-amber-400" :
        "text-slate-400 dark:text-zinc-500";
    uploadStatus.className = `${base} ${toneClass}`;
    uploadStatus.innerText = message;
}

function setMediaStatus(message, tone = "info") {
    if (!mediaStatus) return;
    const base = "mt-3 text-[11px] font-medium uppercase tracking-wider";
    const toneClass =
        tone === "success" ? "text-emerald-600 dark:text-emerald-400" :
        tone === "error" ? "text-red-600 dark:text-red-400" :
        tone === "warn" ? "text-amber-600 dark:text-amber-400" :
        "text-slate-400 dark:text-zinc-500";
    mediaStatus.className = `${base} ${toneClass}`;
    mediaStatus.innerText = message;
}

function setChatIdDisplay(id) {
    if (!chatIdDisplay) return;
    chatIdDisplay.textContent = id ? ("Chat: " + id) : "Chat: —";
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "";
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    return `${mb.toFixed(1)} MB`;
}

function renderMediaList(items) {
    if (!mediaList) return;
    mediaList.innerHTML = "";

    if (!items || items.length === 0) {
        const li = document.createElement("li");
        li.className = "text-xs text-slate-400 dark:text-zinc-500";
        li.textContent = "No media in this chat yet.";
        mediaList.appendChild(li);
        return;
    }

    for (const item of items) {
        const li = document.createElement("li");
        li.className = "flex items-center justify-between gap-2 rounded-xl border border-slate-200 dark:border-white/10 bg-white/60 dark:bg-white/[0.03] px-3 py-2";

        const left = document.createElement("div");
        left.className = "min-w-0";

        const name = document.createElement("div");
        name.className = "text-xs font-semibold text-slate-700 dark:text-zinc-200 truncate";
        name.textContent = item.filename || item.id || "attachment";

        const meta = document.createElement("div");
        meta.className = "text-[10px] font-medium text-slate-400 dark:text-zinc-500 uppercase tracking-wider";
        meta.textContent = `${item.media_type || "media"} • ${formatBytes(item.bytes)}`;

        left.appendChild(name);
        left.appendChild(meta);

        li.appendChild(left);
        mediaList.appendChild(li);
    }
}

async function refreshMediaList() {
    if (!chatId) {
        renderMediaList([]);
        return;
    }
    try {
        const res = await fetch(`/api/media/list?chat_id=${encodeURIComponent(chatId)}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || "Failed to list media");
        renderMediaList(data.attachments || []);
    } catch (_) {
        renderMediaList([]);
    }
}

function setManualInputEnabled(enabled) {
    if (textInput) textInput.disabled = !enabled;
    if (sendTextBtn) sendTextBtn.disabled = !enabled;
}

function sendManualText() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        setManualInputEnabled(false);
        return;
    }
    const raw = textInput?.value || "";
    const text = raw.trim();
    if (!text) return;

    if (text.length > 2) stopPlayback();
    document.querySelectorAll('[data-role="bot-active"]').forEach(m => delete m.dataset.role);

    ws.send(text);
    addMessage(text, "user");
    if (textInput) textInput.value = "";
}

// State
let ws;
let audioCtx;
let nextStartTime = 0;
let activeSources = [];
let chatId = null;
let playbackCtx = null;
let audioQueue = [];
let queuedBytes = 0;
let pumpScheduled = false;

const voicesByLocaleGender = {
    "en-US": { feminine: "tiffany", masculine: "matthew" },
    "en-GB": { feminine: "amy", masculine: null },
    "en-AU": { feminine: "olivia", masculine: null },
    "en-IN": { feminine: "kiara", masculine: "arjun" },
    "fr-FR": { feminine: "ambre", masculine: "florian" },
    "it-IT": { feminine: "beatrice", masculine: "lorenzo" },
    "de-DE": { feminine: "tina", masculine: "lennart" },
    "es-US": { feminine: "lupe", masculine: "carlos" },
    "pt-BR": { feminine: "carolina", masculine: "leo" },
    "hi-IN": { feminine: "kiara", masculine: "arjun" },
};

const polyglotVoices = { feminine: "tiffany", masculine: "matthew" };

function setVoiceOptions() {
    const usePolyglot = !!polyglotToggle?.checked;
    const locale = localeSelect?.value || "en-US";
    const gender = voiceGenderSelect?.value || "masculine";

    const mapping = usePolyglot ? polyglotVoices : (voicesByLocaleGender[locale] || voicesByLocaleGender["en-US"]);
    const available = [];
    if (mapping.feminine) available.push(mapping.feminine);
    if (mapping.masculine) available.push(mapping.masculine);

    voiceSelect.innerHTML = "";
    available.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        voiceSelect.appendChild(opt);
    });

    const desired = mapping[gender] || mapping.feminine || mapping.masculine || available[0];
    voiceSelect.value = available.includes(desired) ? desired : available[0];

    // If locale has no masculine/feminine option, lock gender accordingly.
    if (voiceGenderSelect) {
        const hasFem = !!mapping.feminine;
        const hasMasc = !!mapping.masculine;
        if (!hasMasc) {
            voiceGenderSelect.value = "feminine";
            voiceGenderSelect.disabled = true;
        } else if (!hasFem) {
            voiceGenderSelect.value = "masculine";
            voiceGenderSelect.disabled = true;
        } else {
            voiceGenderSelect.disabled = false;
        }
    }

    if (localeSelect) {
        localeSelect.disabled = usePolyglot;
    }
}

// Initialize locale + voice choices
setVoiceOptions();
if (localeSelect) localeSelect.onchange = () => setVoiceOptions();
if (polyglotToggle) polyglotToggle.onchange = () => setVoiceOptions();
if (voiceGenderSelect) voiceGenderSelect.onchange = () => setVoiceOptions();

// Initialize theme toggle
initTheme();
setManualInputEnabled(false);

// Initialize Visualizer
for(let i=0; i<32; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar w-1.5 rounded-full h-2 bg-indigo-500/60 dark:bg-indigo-400/70 opacity-30';
    visualizer.appendChild(bar);
}

// Reset Logic
resetBtn.onclick = async () => {
    if (!chatId) {
        setUploadStatus("Start a chat first", "warn");
        return;
    }
    if (!confirm("Clear all uploaded documents?")) return;
    try {
        const res = await fetch(`/api/knowledge/reset?chat_id=${encodeURIComponent(chatId)}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            setUploadStatus("Knowledge Base Cleared", "success");
        }
    } catch (err) {
        console.error("Reset failed", err);
        setUploadStatus("Reset failed", "error");
    }
};

// File Ingestion
fileInput.onchange = async () => {
    if (!chatId) {
        setUploadStatus("Start a chat first", "warn");
        fileInput.value = "";
        return;
    }
    const file = fileInput.files[0];
    if (!file) return;

    setUploadStatus("Ingesting " + file.name + "...", "info");
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`/api/knowledge/ingest?chat_id=${encodeURIComponent(chatId)}`, { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            setUploadStatus("Done! " + data.chunks + " chunks added.", "success");
        } else {
            throw new Error();
        }
    } catch (err) {
        setUploadStatus("Upload failed", "error");
    }
};

// Media Upload (image/video) for multimodal tools
if (mediaInput) {
    mediaInput.onchange = async () => {
        if (!chatId) {
            setMediaStatus("Start a chat first", "warn");
            mediaInput.value = "";
            return;
        }

        const file = mediaInput.files[0];
        if (!file) return;

        setMediaStatus("Uploading " + file.name + "...", "info");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`/api/media/upload?chat_id=${encodeURIComponent(chatId)}`, { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data?.detail || "Upload failed");
            }
            setMediaStatus(`Uploaded: ${data.attachment.filename} (${data.attachment.media_type})`, "success");
            await refreshMediaList();
        } catch (err) {
            setMediaStatus("Media upload failed", "error");
        } finally {
            mediaInput.value = "";
        }
    };
}

if (clearMediaBtn) {
    clearMediaBtn.onclick = async () => {
        if (!chatId) {
            setMediaStatus("Start a chat first", "warn");
            return;
        }
        if (!confirm("Clear uploaded images/videos for this chat?")) return;
        try {
            const res = await fetch(`/api/media/clear?chat_id=${encodeURIComponent(chatId)}`, { method: "POST" });
            const data = await res.json();
            if (!res.ok) throw new Error(data?.detail || "Clear failed");
            setMediaStatus("Media cleared", "success");
            await refreshMediaList();
        } catch (err) {
            setMediaStatus("Failed to clear media", "error");
        }
    };
}

if (sendTextBtn) {
    sendTextBtn.onclick = (e) => {
        e.preventDefault?.();
        sendManualText();
    };
}

if (textInput) {
    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendManualText();
        }
    });
}

// Chat Helpers
function addMessage(text, role) {
    transcript.querySelector('[data-placeholder="true"]')?.remove();

    let msgDiv;
    if (role === 'bot' && transcript.lastElementChild?.dataset.role === 'bot-active') {
        msgDiv = transcript.lastElementChild;
    } else {
        msgDiv = document.createElement('div');
        msgDiv.className = role === 'bot'
            ? 'msg self-start px-5 py-3 rounded-2xl rounded-bl-none max-w-[85%] border shadow-sm backdrop-blur whitespace-pre-wrap break-words bg-white/70 text-slate-900 border-slate-200/80 dark:bg-zinc-900/60 dark:text-zinc-100 dark:border-white/10'
            : 'msg self-end px-5 py-3 rounded-2xl rounded-br-none max-w-[85%] border shadow-sm whitespace-pre-wrap break-words bg-indigo-600 text-white border-indigo-600/30 dark:bg-indigo-500 dark:border-white/10';
         
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
    audioQueue = [];
    queuedBytes = 0;
    pumpScheduled = false;
    if (playbackCtx) nextStartTime = playbackCtx.currentTime;
}

function schedulePump() {
    if (pumpScheduled) return;
    pumpScheduled = true;
    queueMicrotask(() => {
        pumpScheduled = false;
        pumpAudio();
    });
}

function pumpAudio() {
    if (!playbackCtx) return;

    // Aim for ~100ms blocks to reduce overhead and improve smoothness.
    const bytesPerSample = 2; // int16
    const targetBlockMs = 100;
    const targetBytes = Math.max(2048, Math.floor((playbackCtx.sampleRate * (targetBlockMs / 1000)) * bytesPerSample));

    while (queuedBytes >= targetBytes) {
        const block = new Uint8Array(targetBytes);
        let offset = 0;
        while (offset < targetBytes && audioQueue.length) {
            const head = audioQueue[0];
            const take = Math.min(head.byteLength, targetBytes - offset);
            block.set(head.subarray(0, take), offset);
            offset += take;
            if (take === head.byteLength) {
                audioQueue.shift();
            } else {
                audioQueue[0] = head.subarray(take);
            }
        }
        queuedBytes -= targetBytes;
        playOutputAudio(block.buffer);
    }
}

// WebSocket Setup
startBtn.onclick = async () => {
    if (ws) { location.reload(); return; }

    try {
        const requestedInputRate = parseInt(inputRateSelect.value, 10);
        const requestedOutputRate = parseInt(outputRateSelect.value, 10);
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: requestedInputRate });
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: requestedOutputRate });
        nextStartTime = playbackCtx.currentTime;

        const wsParams = new URLSearchParams();
        wsParams.set('voice', voiceSelect.value);
        wsParams.set('voice_locale', localeSelect.value);
        wsParams.set('polyglot', polyglotToggle?.checked ? '1' : '0');
        wsParams.set('voice_gender', voiceGenderSelect?.value || 'masculine');
        wsParams.set('assistant_lang', assistantLangSelect?.value || 'auto');
        wsParams.set('code_switch', codeSwitchToggle?.checked ? '1' : '0');
        wsParams.set('endpointing', endpointingSelect.value);
        wsParams.set('input_rate', String(audioCtx.sampleRate));
        wsParams.set('output_rate', String(requestedOutputRate));
        wsParams.set('temperature', String(parseFloat(temperatureRange.value)));
        wsParams.set('top_p', String(parseFloat(topPRange.value)));
        wsParams.set('max_tokens', String(parseInt(maxTokensInput.value, 10)));
        wsParams.set('channels', '1');
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws?${wsParams.toString()}`);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            status.innerText = "Listening...";
            connStatus.innerText = "Connected";
            connStatus.className = "text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest";
            connIndicator.className = "w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-emerald-500/30";
            btnText.innerText = "End Session";
            setManualInputEnabled(true);
            startRecording(stream);
        };

        ws.onmessage = async (e) => {
            if (typeof e.data === 'string') {
                const data = JSON.parse(e.data);
                if (data.event?.chatInit?.chatId) {
                    chatId = data.event.chatInit.chatId;
                    setChatIdDisplay(chatId);
                    await refreshMediaList();
                    return;
                }
                if (data.event?.textOutput) {
                    addMessage(data.event.textOutput.content, 'bot');
                    status.innerText = "Assistant is speaking...";
                    if (data.event.textOutput.isFinal) {
                        document.querySelectorAll('[data-role="bot-active"]').forEach(m => delete m.dataset.role);
                        status.innerText = "Listening...";
                    }
                } else if (data.event?.userTranscript) {
                    if (data.event.userTranscript.trim().length > 2) stopPlayback();
                    document.querySelectorAll('[data-role="bot-active"]').forEach(m => delete m.dataset.role);
                    addMessage(data.event.userTranscript, 'user');
                } else if (data.event?.statusUpdate) {
                    status.innerText = data.event.statusUpdate;
                }
            } else {
                // Buffer audio and play in larger blocks for smoother output.
                const chunk = new Uint8Array(e.data);
                audioQueue.push(chunk);
                queuedBytes += chunk.byteLength;
                schedulePump();
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

function playOutputAudio(arrayBuffer) {
    if (!playbackCtx) return;
    if ((arrayBuffer.byteLength % 2) !== 0) return; // int16 alignment guard
    const int16 = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x7FFF;

    const buffer = playbackCtx.createBuffer(1, float32.length, playbackCtx.sampleRate);
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
