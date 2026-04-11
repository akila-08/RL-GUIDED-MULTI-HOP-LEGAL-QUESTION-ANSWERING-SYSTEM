import { useState, useRef, useEffect } from 'react'
import MessageBubble from './components/MessageBubble'
import ChatInput from './components/ChatInput'
import styles from './App.module.css'

const WELCOME = {
  role: 'assistant',
  content: 'Hello! I am your AI Legal Assistant powered by Hierarchical Reinforcement Learning.\n\nAsk me anything about the Indian Constitution — I can handle both simple lookups and complex multi-hop reasoning questions.',
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = async (question) => {
    setError(null)
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.final_answer || 'No answer returned.',
        metadata: {
          complexity_score:    data.complexity_score,
          is_complex:          data.is_complex,
          actions_taken:       data.actions_taken || [],
          sub_questions:       data.sub_questions || [],
          retrieved_articles:  data.retrieved_articles || [],
          sub_answers:         data.sub_answers || [],
          rewards:             data.rewards || {},
          combined_reward:     data.combined_reward,
        }
      }])
    } catch (err) {
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Cannot connect to backend. Make sure the FastAPI server is running: uvicorn chatbot.app:app --host 0.0.0.0 --port 8000')
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }

  const clearChat = () => setMessages([WELCOME])

  return (
    <div className={styles.layout}>
      {/* ── Sidebar ── */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarLogo}>
          <div className={styles.logoIconWrap}>⚖️</div>
          <div>
            <div className={styles.logoTitle}>Legal AI</div>
            <div className={styles.logoSub}>HRL Assistant</div>
          </div>
        </div>
        <div className={styles.sidebarDivider} />
        <div className={styles.sidebarSection}>
          <div className={styles.sidebarLabel}>Architecture</div>
          {[
            { icon: '🧠', label: 'PPO RL Agent' },
            { icon: '✂️', label: 'Question Decomposer' },
            { icon: '🔍', label: 'Hybrid BM25 + Dense Retriever' },
            { icon: '✍️', label: 'Groq LLM Generator' },
            { icon: '🔗', label: 'Answer Combiner' },
          ].map(({ icon, label }) => (
            <div key={label} className={styles.sidebarItem}>
              <span className={styles.sidebarItemIcon}>{icon}</span>
              <span className={styles.sidebarItemLabel}>{label}</span>
            </div>
          ))}
        </div>
        <div className={styles.sidebarDivider} />
        <div className={styles.sidebarSection}>
          <div className={styles.sidebarLabel}>Pipeline</div>
          {['DECOMPOSE', 'RETRIEVE', 'GENERATE', 'COMBINE'].map((step, i) => (
            <div key={step} className={styles.pipelineStep}>
              <div className={styles.pipelineNum}>{i + 1}</div>
              <span className={styles.pipelineLabel}>{step}</span>
            </div>
          ))}
        </div>
        <div className={styles.sidebarSpacer} />
        <button className={styles.clearBtn} onClick={clearChat}>
          🗑️ Clear Chat
        </button>
      </aside>

      {/* ── Main area ── */}
      <main className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <h1 className={styles.headerTitle}>Constitution of India QA</h1>
            <span className={styles.headerSub}>Powered by Hierarchical Reinforcement Learning</span>
          </div>
          <div className={styles.headerRight}>
            <div className={styles.headerBadge}>
              <span className={styles.dot} />
              443 Articles indexed
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className={styles.messages}>
          {messages.map((msg, i) => (
            <MessageBubble key={i} role={msg.role} content={msg.content} metadata={msg.metadata} />
          ))}
          {loading && <MessageBubble role="assistant" isThinking />}
          {error && (
            <div className={styles.errorBanner}>
              ⚠️ {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <ChatInput onSend={handleSend} disabled={loading} />
      </main>
    </div>
  )
}
