import { useState, useEffect, useRef } from "react";

function App() {
  const [strictMode, setStrictMode] = useState(false);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("Detailed");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [historyItems, setHistoryItems] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [analytics, setAnalytics] = useState(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const API = process.env.REACT_APP_API_URL;
  if (!API) {
  console.error("REACT_APP_API_URL is not defined");
}
  const authHeaders = () => ({
  "Content-Type": "application/json",
  "Authorization": `Bearer ${localStorage.getItem("token")}`
});
  useEffect(() => {
  chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages]);
  function clearChat() {
  setMessages([]);
}

async function loadAnalytics() {
  const res = await fetch(`${API}/api/analytics`, {
    headers: authHeaders()
  });

  const data = await res.json();
  setAnalytics(data);
  setShowAnalytics(true);
}
  
function convertMarkdownBold(text) {
  return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function highlightSources(answer, chunks) {
  let safe = answer;

  // Convert markdown bold
  safe = convertMarkdownBold(safe);

  // Escape everything except our allowed tags
  safe = safe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/&lt;(\/?(mark|strong))&gt;/g, "<$1>");

  if (!chunks || chunks.length === 0) {
    return safe.replace(/\n/g, "<br/>");
  }

  chunks.forEach(chunk => {
    if (!chunk || chunk.length < 20) return;

    const cleanChunk = chunk.replace(/[*_`#]/g, "");

    const escaped = cleanChunk
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      .replace(/\s+/g, "\\s+");

    const regex = new RegExp(`(${escaped})`, "gi");

    safe = safe.replace(regex, `<mark>$1</mark>`);
  });

  return safe.replace(/\n/g, "<br/>");
}

async function sendFeedback(messageId, feedback) {
  try {
    await fetch(`${API}/api/feedback`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ id: messageId, feedback })
    });

    setMessages(prev =>
      prev.map(m =>
        m.id === messageId ? { ...m, feedback } : m
      )
    );
  } catch (err) {
    console.error("Feedback failed", err);
  }
}

async function toggleBookmark(id, value) {
  await fetch(`${API}/api/bookmark`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ id, value })
  });

  // Update chat messages
  setMessages(prev =>
    prev.map(m =>
      m.id === id ? { ...m, bookmarked: value } : m
    )
  );

  // Update history panel items
  setHistoryItems(prev =>
    prev.map(item =>
      item._id === id ? { ...item, bookmarked: value } : item
    )
  );

  // Refresh analytics if open
  if (showAnalytics) {
    loadAnalytics();
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
      headers: {
        "Authorization": `Bearer ${localStorage.getItem("token")}`
      },
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
  const res = await fetch(`${API}/api/history/id/${id}`, {
    headers: authHeaders()
  });

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
      id: data._id,                
      role: "bot",
      text: highlightSources(data.text, data.chunks),
      confidence: data.confidence || "Unknown",
      coverage: botCoverage,
      sources: data.sources || [],
      feedback: data.feedback || null
    }
  ]);

  setShowHistoryPanel(false);
}


async function loadHistoryPanel() {
  const res = await fetch(`${API}/api/history`, {
    headers: authHeaders()
  });
  const data = await res.json();
  setHistoryItems(data);
  setShowHistoryPanel(true);
}

async function deleteHistoryItem(q) {
  if (!window.confirm("Delete this item?")) return;

  await fetch(`${API}/api/history/${encodeURIComponent(q)}`, {
    method: "DELETE",
    headers: authHeaders()
  });

  setHistoryItems(prev => prev.filter(item => item.question !== q));
}

async function deleteAllHistory() {
  if (!window.confirm("Delete ALL history?")) return;

  await fetch(`${API}/api/history`, {
    method: "DELETE",
    headers: authHeaders()
  });

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
      headers: authHeaders(),
      body: JSON.stringify({ question, mode, memory, strict: strictMode })
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
      id: data.id,
      role: "bot",
      text: highlightSources(data.text, data.chunks),
      confidence: data.confidence,
      coverage: data.coverage,
      sources: data.sources,
      feedback: null,
      bookmarked: false      
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
  if (!localStorage.getItem("token")) {
    return <Login onLogin={() => window.location.reload()} />;
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

        <label style={{ marginLeft: "10px", fontSize: "14px" }}>
          <input
            type="checkbox"
            checked={strictMode}
            onChange={() => setStrictMode(!strictMode)}
          /> Strict mode (only answer from docs)
        </label>

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
        
        <button 
          onClick={loadAnalytics} 
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
          Show Analytics
        </button>

        <button
          onClick={() => window.open(`${API}/api/export?token=${localStorage.getItem("token")}`, "_blank")}
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
          Export PDF
        </button>


        <input 
          type="file"
          onChange={uploadFile}
          disabled={uploading}
          style={{marginLeft: "10px"}}
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
              {m.role === "bot" ? (
                <div dangerouslySetInnerHTML={{ __html: m.text }} />
              ) : (m.text
              )}
              {m.role === "bot" && (
                <div style={{ fontSize: "12px", marginTop: "4px", opacity: 0.6 }}>
                  {m.confidence === "Strict mode" && (
                    <div style={{ color: "red", fontWeight: "bold", fontSize: "12px" }}>
                      ⚠ Strict mode blocked this answer
                    </div>
                  )}

                {m.feedback && (
                  <div style={{
                    marginTop: "4px",
                    fontSize: "12px",
                    color: m.feedback === "up" ? "green" : "red",
                    fontWeight: "bold"
                  }}>
                    {m.feedback === "up" ? "✔ Marked helpful" : "✖ Marked wrong"}
                  </div>
                )}
                <div>
                  {m.confidence} |{" "}
                  {m.coverage && typeof m.coverage === "object"
                    ? `🧠 Grounded: ${m.coverage.grounded ?? 0}% | General: ${m.coverage.general ?? 0}%`
                    : m.coverage !== undefined
                    ? `Coverage: ${m.coverage}%`
                    : "Coverage: N/A"}
                </div>
                <div style={{ marginTop: "4px" }}>
                  <button onClick={() => toggleBookmark(m.id, !m.bookmarked)}>
                    {m.bookmarked ? "⭐ Bookmarked" : "☆ Bookmark"}
                  </button>
                  <button onClick={() => sendFeedback(m.id, "up")}>👍</button>
                  <button onClick={() => sendFeedback(m.id, "down")} style={{ marginLeft: "6px" }}>👎</button>
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
          <input
            type="text"
            placeholder="Search history..."
            onChange={async (e) => {
              const q = e.target.value;
              if (!q) {
                loadHistoryPanel();
                return;
              }

              const res = await fetch(`${API}/api/history/search?q=${encodeURIComponent(q)}`, {
                headers: authHeaders()
              });
              const data = await res.json();
              setHistoryItems(data);
            }}
            style={{
              width: "100%",
              padding: "6px",
              marginBottom: "10px",
              border: "1px solid #ddd",
              borderRadius: "4px"
            }}
          />
          <label style={{ fontSize: "12px" }}>
            <input
              type="checkbox"
              onChange={(e) => {
                if (e.target.checked) {
                  setHistoryItems(prev => prev.filter(i => i.bookmarked));
                } else {
                  loadHistoryPanel();
                }
              }}
            /> Show only bookmarked
          </label>
  
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
                style={{ cursor: "pointer", flex: 1 }}
                onClick={() => loadHistoryItem(item._id)}
              >
                {item.bookmarked ? "⭐ " : ""}{item.question}
              </span>
                
              <button 
                onClick={() => toggleBookmark(item._id, !item.bookmarked)}
                style={{ marginRight: "6px" }}
              >
                {item.bookmarked ? "⭐ Bookmarked" : "☆ Bookmark"}
              </button>
                
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
      {showAnalytics && analytics && (
        <div style={{
          position: "fixed",
          left: 0,
          top: 0,
          height: "100vh",
          width: "300px",
          background: "white",
          borderRight: "1px solid #ddd",
          padding: "16px",
          overflowY: "auto",
          zIndex: 1000
        }}>
          <h3>Analytics</h3>
          <p>Total Q&A: {analytics.total}</p>
          <p>👍 Helpful: {analytics.helpful}</p>
          <p>👎 Wrong: {analytics.wrong}</p>
          <p>⭐ Bookmarked: {analytics.bookmarked}</p>
          <p>Hallucination rate: {analytics.hallucination_rate}%</p>

          <h4>Top Questions</h4>
          <ul>
            {analytics.top_questions.map((q, i) => (
              <li key={i}>{q._id} ({q.count})</li>
            ))}
          </ul>

            <h4>Top Sources</h4>
            <ul style={{ paddingLeft: "18px" }}>
              {analytics.top_sources.map((s, i) => (
                <li
                  key={i}
                  style={{
                    wordBreak: "break-all",
                    whiteSpace: "normal",
                    fontSize: "13px",
                    marginBottom: "6px"
                  }}
                  title={s._id}
                >
                  {s._id} ({s.count})
                </li>
              ))}
            </ul>
            <button onClick={() => setShowAnalytics(false)}>Close</button>
          </div>
        )}
    </div>
  );
}

export default App;
