import { useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [question, setQuestion] = useState('');
  const [pipeline, setPipeline] = useState('raw');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, pipeline }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Something went wrong');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Islam QnA — RAG </h1>
      <p className="subtitle">
        Ask a question about Islam
      </p>

      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What are the Five Pillars of Islam?"
          className="question-input"
        />

        <div className="pipeline-selector">
          <label>
            <input
              type="radio"
              value="raw"
              checked={pipeline === 'raw'}
              onChange={(e) => setPipeline(e.target.value)}
            />
            Raw Pipeline  
          </label>
          <label>
            <input
              type="radio"
              value="langchain"
              checked={pipeline === 'langchain'}
              onChange={(e) => setPipeline(e.target.value)}
            />
            LangChain
          </label>
          <label>
            <input
              type="radio"
              value="langgraph"
              checked={pipeline === 'langgraph'}
              onChange={(e) => setPipeline(e.target.value)}
            />
            LangGraph
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Thinking.....' : 'Ask'}
        </button>
      </form>

      {error && <div className="error">Error: {error}</div>}

      {result && (
        <div className="result">
          <h3>Answer</h3>
          <p className="answer-text">{result.answer}</p>

          <div className="meta">
            <span><strong>Pipeline:</strong> {result.pipeline_used}</span>
            <span><strong>Response time:</strong> {result.response_time_seconds}s</span>
            {result.retries_used !== null && (
              <span><strong>Retries used:</strong> {result.retries_used}</span>
            )}
            {result.is_grounded !== null && (
              <span><strong>Grounded:</strong> {result.is_grounded ? 'Yes' : 'No'}</span>
            )}
          </div>

          <h4>Sources</h4>
          <ul className="sources-list">
            {result.sources.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;