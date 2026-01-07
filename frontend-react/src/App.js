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

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (!API) {
    return (
      <div style={{ padding: 40, color: "red", textAlign: "center" }}>
        ❌ REACT_APP_API_URL is not defined. Set it in your environment.
      </div>
    );
  }

  function clearChat() {
    setMessages([]);
  }

  async function uploadFile(e) {
    const file = e.target.files[0];
    if (!file || uploading) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
      const text = await res.text();

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text || "Invalid server response");
      }

      if (!res.ok) throw new Error(data.error || "Upload failed");

      alert(data.message || "Uploaded successfully");
    } catch (err) {
      alert(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function loadHistoryItem(q) {
    try {
      const res = await fetch(`${API}/api/history/${encodeURIComponent(q)}`);
      const data = await res.json();

      setMessages([
        { role: "user", text: data.question },
        { role: "bot", text: data.text, confidence: data.confidence, coverage: data.coverage }
      ]);
      setShowHistoryPanel(false);
    } catch (err) {
      alert("Failed to load history item");
    }
  }

  async function loadHistoryPanel() {
    try {
      const res = await fetch(`${API}/api/history`);
      const data = await res.json();
      setHistoryItems(data);
      setShowHistoryPanel(true);
    } catch {
      alert("Failed to load history");
    }
  }

  async function ask() {
    if (!question.trim() || loading) return;

    const userMessage = { role: "user", text: question };
    const updatedMessages = [...messages, userMessage];

    setMessages(updatedMessages);
    setQuestion("");
    setLoading(true);

    try {
      const memory = updatedMessages.slice(-6);

      const res = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode, memory })
      });

      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text || "Invalid server response");
      }

      if (!res.ok) throw new Error(data.error || "Server error");

      const botMessage = {
        role: "bot",
        text: data.text,
        confidence: data.confidence,
        coverage: data.coverage
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setMessages(prev => [...prev, { role: "bot", text: `⚠️ ${err.message || "Error contacting server"}` }]);
    } finally {
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
    <div style={{ minHeight: "100vh", display: "flex", justifyContent: "center", background: "#f3f4f6", paddingTop: 40 }}>
      <div style={{ background: "white", padding: 30, borderRadius: 12, width: "100%", maxWidth: 700 }}>
        <h2 style={{ textAlign: "center" }}>RAG Chatbot</h2>

        <textarea rows={4} value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={handleKeyDown}
          placeholder="Ask your question..." style={{ width: "100%", padding: 12, borderRadius: 8 }} />

        <br /><br />

        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option>Concise</option>
          <option>Detailed</option>
          <option>Exam</option>
          <option>ELI5</option>
          <option>Compare</option>
        </select>

        <br /><br />

        <button onClick={ask} disabled={loading || uploading}>{loading ? "Thinking..." : "Ask"}</button>
        <button onClick={clearChat} style={{ marginLeft: 8 }}>Clear</button>
        <button onClick={loadHistoryPanel} style={{ marginLeft: 8 }}>Show History</button>
        <input type="file" onChange={uploadFile} disabled={uploading} style={{ marginLeft: 8 }} />

        <div style={{ marginTop: 20 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ marginBottom: 8, textAlign: m.role === "user" ? "right" : "left" }}>
              <div style={{ display: "inline-block", padding: 10, borderRadius: 8, background: m.role === "user" ? "#2563eb" : "#e5e7eb", color: m.role === "user" ? "white" : "black" }}>
                {m.role === "bot" ? <ReactMarkdown>{m.text}</ReactMarkdown> : m.text}
                {m.role === "bot" && <div style={{ fontSize: 12 }}>{m.confidence} | Coverage: {m.coverage}%</div>}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
      </div>

      {showHistoryPanel && (
        <div style={{ position: "fixed", right: 0, top: 0, width: 300, height: "100vh", background: "white", padding: 16 }}>
          <h3>History</h3>
          <button onClick={() => setShowHistoryPanel(false)}>Close</button>
          {historyItems.map((item, i) => (
            <div key={i} style={{ cursor: "pointer", padding: 6 }} onClick={() => loadHistoryItem(item.question)}>
              {item.question}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
