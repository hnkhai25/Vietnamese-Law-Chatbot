import React, { useState } from "react";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode]   = useState("hybrid"); // semantic | bm25 | hybrid | hybrid_rerank
  const [k, setK]         = useState(3);
  const [messages, setMessages] = useState([]); // {role:'user'|'bot', text:string, contexts?:[]}
  const [loading, setLoading]   = useState(false);

  const sendQuery = async (e) => {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    // push user message
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setQuery("");
    setLoading(true);

    try {
      // gọi API /generate (có mode & k)
      const res = await axios.post(`${API_BASE}/generate`, { query: q, k, mode });
      const data = res.data;

      const answer = data.answer || "Không có phản hồi.";
      const contexts = data.contexts || [];

      setMessages(prev => [...prev, { role: "bot", text: answer, contexts }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: "bot", text: "❌ Lỗi kết nối server." }]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h2>⚖️ Vietnamese Law Chatbot</h2>

      <div className="toolbar">
        <label>
          Mode:&nbsp;
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="semantic">semantic</option>
            <option value="bm25">bm25</option>
            <option value="hybrid">hybrid</option>
            <option value="hybrid_rerank">hybrid_rerank</option>
          </select>
        </label>

        <label>
          Top-k:&nbsp;
          <input
            type="number"
            min="1"
            max="10"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
      </div>

      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className="avatar">{m.role === "user" ? "🧑‍💼" : "🤖"}</div>
            <div className="bubble">
              <div className="text">{m.text}</div>

              {m.role === "bot" && m.contexts && m.contexts.length > 0 && (
                <details className="contexts">
                  <summary>📎 Nguồn trích (contexts)</summary>
                  <ul>
                    {m.contexts.map((c, idx) => (
                      <li key={idx}>
                        <div className="ctx-text">• {c.text}</div>
                        <div className="ctx-meta">
                          <small>
                            doc_id: {c.doc_id} | sem: {Number(c.score_semantic ?? 0).toFixed(3)}
                            {c.score_bm25 !== undefined ? ` | bm25: ${Number(c.score_bm25).toFixed(3)}` : ""}
                            {c.score_hybrid !== undefined ? ` | hybrid: ${Number(c.score_hybrid).toFixed(3)}` : ""}
                          </small>
                        </div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && <div className="chat-msg bot"><div className="avatar">🤖</div><div className="bubble">⏳ Đang tạo câu trả lời…</div></div>}
      </div>

      <form className="chat-form" onSubmit={sendQuery}>
        <textarea
          placeholder="Nhập câu hỏi pháp luật của bạn…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={3}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          Gửi
        </button>
      </form>

      <footer className="footer">
        <small>API: {API_BASE} &middot; Mode: {mode} &middot; k={k}</small>
      </footer>
    </div>
  );
}
