import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

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

function normalize(text) {
  return text
    .toLowerCase()
    .replace(/[*_`~#>\-\n]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function highlightSources(answer, chunks) {
  if (!chunks || chunks.length === 0) return answer;

  let result = answer;

  chunks.forEach(chunk => {
    if (!chunk || chunk.length < 20) return;

    const cleanChunk = chunk.replace(/[*_`#]/g, "");

    const escaped = cleanChunk
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      .replace(/\s+/g, "\\s+");

    const regex = new RegExp(`(${escaped})`, "gi");

    result = result.replace(
      regex,
      `<mark style="background:#d1fae5;padding:2px 4px;border-radius:4px">$1</mark>`
    );
  });

  return result;
}

async function sendFeedback(text, feedback) {
  try {
    await fetch(`${API}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, feedback })
    });
  } catch (err) {
    console.error("Feedback failed", err);
  }
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


async function loadHistoryItem(id) {
  const res = await fetch(`${API}/api/history/id/${id}`);
  const data = await res.json();

  const botCoverage =
    data.coverage && typeof data.coverage === "object"
      ? data.coverage
      : data.coverage !== undefined
      ? { grounded: data.coverage, general: 100 - data.coverage }
      : { grounded: 0, general: 0 };

  setMessages([
    { role: "user", text: data.question },
    {
      role: "bot",
      text: data.text || "No answer found.",
      confidence: data.confidence || "Unknown",
      coverage: botCoverage,
      sources: data.sources || []
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

async function deleteHistoryItem(q) {
  if (!window.confirm("Delete this item?")) return;

  await fetch(`${API}/api/history/${encodeURIComponent(q)}`, {
    method: "DELETE"
  });

  setHistoryItems(prev => prev.filter(item => item.question !== q));
}

async function deleteAllHistory() {
  if (!window.confirm("Delete ALL history?")) return;

  await fetch(`${API}/api/history`, { method: "DELETE" });
  setHistoryItems([]);
  setMessages([]);
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

    console.log("Received chunks:", data.chunks);
    const botMessage = {
      role: "bot",
      text: highlightSources(data.text, data.chunks),
      confidence: data.confidence,
      coverage: data.coverage,
      sources: data.sources
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
              {m.role === "bot" ? <ReactMarkdown rehypePlugins={[rehypeRaw]}>{m.text}</ReactMarkdown> : m.text}
              {m.role === "bot" && (
                <div style={{ fontSize: "12px", marginTop: "4px", opacity: 0.6 }}>
                <div>
                  {m.confidence} |{" "}
                  {m.coverage && typeof m.coverage === "object"
                    ? `🧠 Grounded: ${m.coverage.grounded ?? 0}% | General: ${m.coverage.general ?? 0}%`
                    : m.coverage !== undefined
                    ? `Coverage: ${m.coverage}%`
                    : "Coverage: N/A"}
                </div>
                <div style={{ marginTop: "4px" }}>
                  <button onClick={() => sendFeedback(m.text, "up")}>👍</button>
                  <button onClick={() => sendFeedback(m.text, "down")} style={{ marginLeft: "6px" }}>👎</button>
                </div>
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
          <div style={{ marginBottom: "8px" }}>
            <button onClick={() => setShowHistoryPanel(false)}>Close</button>
            <button 
              onClick={deleteAllHistory}
              style={{ marginLeft: "8px", color: "red" }}
            >
              Delete All
            </button>
          </div>

          {historyItems.map((item, i) => (
            <div
              key={item._id}
              style={{
                padding: "8px",
                borderBottom: "1px solid #eee",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center"
              }}
            >
              <span
                style={{ cursor: "pointer" }}
                onClick={() => loadHistoryItem(item._id)}
              >
                {item.question}
              </span>
              <button
                onClick={() => deleteHistoryItem(item.question)}
                style={{
                  background: "none",
                  border: "none",
                  color: "red",
                  cursor: "pointer"
                }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
