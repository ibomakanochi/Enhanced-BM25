import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [contextDocs, setContextDocs] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) return;
    
    // Reset state for new query
    setAnswer('');
    setContextDocs([]);
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/retrieve_and_generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      // Continuously read the stream chunks
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || ""; // Keep incomplete chunk in buffer

        for (const event of events) {
          if (event.startsWith("data: ")) {
            const dataStr = event.slice(6);
            
            try {
              const data = JSON.parse(dataStr);

              if (data.type === "context") {
                // Instantly show the Top 3 scientific documents retrieved by BM25+CrossEncoder
                setContextDocs(data.retrieved);
              } else if (data.type === "token") {
                // Append the LLM word to the UI to create the typing effect
                setAnswer((prev) => prev + data.text);
              } else if (data.type === "done" || data.type === "error") {
                if (data.type === "error") setAnswer(data.text);
                setLoading(false);
              }
            } catch (err) {
              console.warn("Failed to parse chunk:", dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error connecting to backend:", error);
      setAnswer("Failed to connect to the backend server.");
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>Enhanced Two-Stage RAG System</h2>
      
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <input 
          type="text" 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Ask a question about the documents..."
          style={{ flex: 1, padding: '10px', fontSize: '16px', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button 
          onClick={handleSearch} 
          disabled={loading && !answer} 
          style={{ padding: '10px 20px', cursor: 'pointer', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          {loading && !answer ? 'Retrieving...' : 'Search'}
        </button>
      </div>

      {(answer || loading) && (
        <div style={{ padding: '15px', backgroundColor: '#f0f8ff', borderRadius: '5px', marginBottom: '20px', minHeight: '80px' }}>
          <strong>Generated Answer:</strong>
          <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
            {answer}
            {loading && <span style={{ color: '#007bff', animation: 'blink 1s step-end infinite' }}> ▋</span>}
          </p>
        </div>
      )}
      
      {contextDocs.length > 0 && (
        <div>
          <h3>Retrieved Context (Cross-Encoder Re-ranked)</h3>
          {contextDocs.map((doc, idx) => (
            <div key={idx} style={{ padding: '10px', border: '1px solid #ddd', marginBottom: '10px', borderRadius: '5px' }}>
              <span style={{ color: 'gray', fontSize: '12px', fontWeight: 'bold' }}>Neural Score: {doc.score.toFixed(4)}</span>
              <p style={{ margin: '5px 0', fontSize: '14px', lineHeight: '1.4' }}>{doc.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;