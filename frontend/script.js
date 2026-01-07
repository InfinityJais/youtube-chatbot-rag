// --- Configuration ---
const CHAT_HISTORY = document.getElementById('chatHistory');
const VIDEO_CONTAINER = document.getElementById('videoContainer');
const VIDEO_PLACEHOLDER = document.getElementById('videoPlaceholder');
const CHAT_INPUT = document.getElementById('chatInput');
const SEND_BUTTON = document.getElementById('sendButton');

// Placeholder for your FastAPI backend URL
const API_URL = "http://127.0.0.1:8000"; 
let currentVideoId = null;

// --- Utility Functions ---

/** Extracts the video ID from various YouTube URL formats. */
function extractVideoId(url) {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/i;
    const match = url.match(regex);
    return (match && match[1].length === 11) ? match[1] : null;
}

/** Scrolls the chat history to the bottom. */
function scrollToBottom() {
    CHAT_HISTORY.scrollTop = CHAT_HISTORY.scrollHeight;
}

/** Adds a message bubble to the chat history. */
function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    
    const p = document.createElement('p');
    p.innerText = text;
    messageDiv.appendChild(p);
    
    CHAT_HISTORY.appendChild(messageDiv);
    scrollToBottom();
}

/** Handles keypress events for the chat input. */
CHAT_INPUT.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});


// --- Core Functions ---

/** Loads the YouTube video into the player. */
function loadVideo() {
    const url = document.getElementById('youtubeUrl').value.trim();
    const videoId = extractVideoId(url);

    if (!videoId) {
        alert("Please enter a valid YouTube URL.");
        return;
    }

    currentVideoId = videoId;

    // 1. Construct the iframe HTML
    const iframeHTML = `<iframe 
        src="https://www.youtube.com/embed/${videoId}" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>`;

    // 2. Inject and show the video container
    VIDEO_CONTAINER.innerHTML = iframeHTML;
    VIDEO_CONTAINER.classList.remove('hidden');
    VIDEO_PLACEHOLDER.style.display = 'none';

    // 3. Optional: Trigger your backend ingestion API here
    // We'll simulate the ingestion step with a message
    addMessage("Video loaded. Preparing RAG data... This usually takes a moment.", false);
    
    // --- REAL WORLD: CALL YOUR FASTAPI INGESTION ENDPOINT ---
    // Example: fetch(`${API_URL}/ingest-youtube/${videoId}`)
    // .then(response => { if (response.ok) addMessage("RAG data ready! Ask me anything.", false); });
}

/** Sends the user message to the RAG API and displays the response. */
async function sendMessage() {
    const userQuery = CHAT_INPUT.value.trim();
    if (!userQuery) return;
    
    if (!currentVideoId) {
        addMessage("Please load a YouTube video first!", false);
        return;
    }

    // 1. Display user message and clear input
    addMessage(userQuery, true);
    CHAT_INPUT.value = '';
    SEND_BUTTON.disabled = true;
    CHAT_INPUT.placeholder = 'Waiting for response...';
    
    const thinkingMessage = addThinkingIndicator(); // Show indicator

    try {
        // --- REAL WORLD: CALL YOUR FASTAPI QUERY ENDPOINT ---
        const response = await fetch(`${API_URL}/query-rag/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: userQuery,
                index_name: "your-pinecone-index-name" // <-- REPLACE WITH YOUR INDEX NAME
            }),
        });

        CHAT_HISTORY.removeChild(thinkingMessage); // Remove indicator

        if (!response.ok) {
            const errorData = await response.json();
            addMessage(`Error: ${errorData.detail || 'Could not connect to RAG API.'}`, false);
            return;
        }

        const data = await response.json();
        addMessage(data.answer || "Sorry, I couldn't find an answer.", false);

    } catch (error) {
        CHAT_HISTORY.removeChild(thinkingMessage); // Remove indicator on catch
        console.error('API Error:', error);
        addMessage("Connection error. Ensure the FastAPI server is running.", false);
    } finally {
        SEND_BUTTON.disabled = false;
        CHAT_INPUT.placeholder = 'Ask a question about the video...';
    }
}

/** Adds a temporary typing/thinking indicator. */
function addThinkingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.classList.add('message', 'bot-message', 'thinking-indicator');
    indicatorDiv.innerHTML = '<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
    
    // Add simple CSS for the dots (could be moved to style.css)
    const style = document.createElement('style');
    style.innerHTML = `
        .thinking-indicator .dot {
            display: inline-block;
            animation: thinking 1.5s infinite steps(1, start);
            opacity: 0;
        }
        .thinking-indicator .dot:nth-child(1) { animation-delay: 0s; }
        .thinking-indicator .dot:nth-child(2) { animation-delay: 0.5s; }
        .thinking-indicator .dot:nth-child(3) { animation-delay: 1.0s; }
        @keyframes thinking {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    CHAT_HISTORY.appendChild(indicatorDiv);
    scrollToBottom();
    return indicatorDiv;
}