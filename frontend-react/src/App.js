import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";


function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("Detailed");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [uploading, setUploading] = useState(false);
  const API = process.env.REACT_APP_API_URL;
  if (!API) {
  console.error("REACT_APP_API_URL is not defined");
}
  useEffect(() => {
  chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages]);
  function clearChat() {
  setMessages([]);
}

async function uploadFile(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  setUploading(true);
  try {
    const res = await fetch(`${API}/api/upload`, {
      method: "POST",
      body: formData
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error("Invalid server response");
    }
    if (!res.ok) {
      throw new Error(data.error || "Upload failed");
    }
    alert(data.message || "Uploaded");
  } catch (err) {
    alert("Upload failed");  
  } finally {
    setUploading(false);
  }
}


async function loadHistoryItem(q) {
  const res = await fetch(`${API}/api/history/${encodeURIComponent(q)}`);
  const data = await res.json();

  setMessages([
    { role: "user", text: data.question },
    {
      role: "bot",
      text: data.text,
      confidence: data.confidence,
      coverage: data.coverage
    }
  ]);

  setShowHistoryPanel(false);
}

async function loadHistoryPanel() {
  const res = await fetch(`${API}/api/history`);
  const data = await res.json();
  setHistoryItems(data);
  setShowHistoryPanel(true);
}
  

async function ask() {
  if (!question.trim() || loading) return;

  setLoading(true);

  const userMessage = { role: "user", text: question };
  setMessages(prev => [...prev, userMessage]);
  setQuestion("");

  try {
    const memory = messages.slice(-6);

    const res = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, memory })
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error("Invalid server response");
    }
    if (!res.ok) {
      throw new Error(data.error || "Server error");
    }


    const botMessage = {
      role: "bot",
      text: data.text,
      confidence: data.confidence,
      coverage: data.coverage
    };

    setMessages(prev => [...prev, botMessage]);
  } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        { role: "bot", text: `⚠️ ${err.message || "Error contacting server."}` }
      ]);
  }
 finally {
    setLoading(false);
  }
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
}

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      justifyContent: "center",
      alignItems: "flex-start",
      background: "#f3f4f6",
      paddingTop: "40px"
    }}>
      <div style={{
        background: "white",
        padding: "30px",
        borderRadius: "12px",
        width: "100%",
        maxWidth: "700px",
        boxShadow: "0 10px 20px rgba(0,0,0,0.08)"
      }}>
        <h2 style={{ textAlign: "center" }}>RAG Chatbot</h2>

        <textarea
          rows={4}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask your question..."
          style={{
            width: "100%",
            padding: "12px",
            fontSize: "16px",
            borderRadius: "8px",
            border: "1px solid #ddd",
            outline: "none"
          }}
        />

        <br /><br />

        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          style={{
            padding: "8px",
            borderRadius: "6px",
            border: "1px solid #ddd"
          }}
        >
          <option>Concise</option>
          <option>Detailed</option>
          <option>Exam</option>
          <option>ELI5</option>
          <option>Compare</option>
        </select>

        <br /><br />

        <button
          onClick={ask} 
          disabled={loading || uploading}
          style={{
            padding: "10px 20px",
            fontSize: "15px",
            borderRadius: "6px",
            border: "none",
            background: "#2563eb",
            color: "white",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
        <button
          onClick={clearChat}
          style={{
            marginLeft: "10px",
            padding: "10px 20px",
            fontSize: "15px",
            borderRadius: "6px",
            border: "1px solid #ddd",
            background: "white",
            cursor: "pointer"
          }}
        >
          Clear
        </button>
        <button 
          onClick={loadHistoryPanel}
          style={{
            marginLeft: "10px",
            padding: "10px 20px",
            fontSize: "15px",
            borderRadius: "6px",
            border: "1px solid #ddd",
            background: "white",
            cursor: "pointer"
          }}
        >
          Show History
        </button>

        <input 
          type="file"
          onChange={uploadFile}
          disabled={uploading}
          style={{ marginLeft: "10px" }}
        />

        {uploading && <span style={{ marginLeft: "10px" }}>Uploading...</span>}

        <div style={{ marginTop: "24px" }}>
          {messages.map((m, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              justifyContent: m.role === "user" ? "flex-end" : "flex-start",
              marginBottom: "10px"
            }}
          >
            <div
              style={{
                padding: "10px 14px",
                borderRadius: "12px",
                maxWidth: "80%",
                background: m.role === "user" ? "#2563eb" : "#e5e7eb",
                color: m.role === "user" ? "white" : "black"
              }}
            >
              {m.role === "bot" ? <ReactMarkdown>{m.text}</ReactMarkdown> : m.text}
              {m.role === "bot" && (
                <div style={{ fontSize: "12px", marginTop: "4px", opacity: 0.6 }}>
                  {m.confidence} | Coverage: {m.coverage}%
                </div>
              )}
              {m.sources && m.sources.length > 0 && (
                <div style={{ marginTop: "6px", fontSize: "12px" }}>
                  <b>Sources:</b>
                  <ul>
                    {m.sources.map((s, i) => (
                      <li key={i}>
                        {s.source} (page {s.page})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: "10px" }}>
            <div style={{
              padding: "10px 14px",
              borderRadius: "12px",
              background: "#e5e7eb",
              color: "black",
              fontStyle: "italic"
            }}>
              Bot is typing...
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      </div>
      {showHistoryPanel && (
        <div style={{
          position: "fixed",
          right: 0,
          top: 0,
          height: "100vh",
          width: "300px",
          background: "white",
          borderLeft: "1px solid #ddd",
          boxShadow: "-4px 0 10px rgba(0,0,0,0.05)",
          padding: "16px",
          overflowY: "auto",
          zIndex: 1000
        }}>
          <h3>History</h3>
          <button onClick={() => setShowHistoryPanel(false)}>Close</button>

          {historyItems.map((item, i) => (
            <div
              key={i}
              style={{
                padding: "8px",
                borderBottom: "1px solid #eee",
                cursor: "pointer"
              }}
              onClick={() => loadHistoryItem(item.question)}
            >
              {item.question}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;

this is my complete app.js if anything present unecessarily fix it and replace it
