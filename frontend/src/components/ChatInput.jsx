import { useRef, useEffect } from 'react'
import styles from './ChatInput.module.css'

export default function ChatInput({ onSend, disabled }) {
  const textareaRef = useRef(null)

  useEffect(() => {
    if (!disabled && textareaRef.current) textareaRef.current.focus()
  }, [disabled])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const submit = () => {
    const val = textareaRef.current?.value?.trim()
    if (!val || disabled) return
    onSend(val)
    textareaRef.current.value = ''
    textareaRef.current.style.height = 'auto'
  }

  const autoResize = () => {
    const t = textareaRef.current
    if (!t) return
    t.style.height = 'auto'
    t.style.height = Math.min(t.scrollHeight, 120) + 'px'
  }

  const suggestions = [
    'What is Article 1?',
    'Difference between Article 14 and 21?',
    'How do Articles 14 and 16(2) ensure fairness?',
    'Who appoints the Chief Justice of India?',
  ]

  return (
    <div className={styles.container}>
      <div className={styles.suggestions}>
        {suggestions.map((s) => (
          <button key={s} className={styles.chip} disabled={disabled}
            onClick={() => { textareaRef.current.value = s; autoResize(); submit() }}>
            {s}
          </button>
        ))}
      </div>
      <div className={`${styles.inputRow} ${disabled ? styles.inputDisabled : ''}`}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder="Ask about the Indian Constitution…  (Enter to send, Shift+Enter for newline)"
          onKeyDown={handleKeyDown}
          onInput={autoResize}
          rows={1}
          disabled={disabled}
        />
        <button className={`${styles.sendBtn} ${disabled ? styles.sendBtnDisabled : ''}`}
          onClick={submit} disabled={disabled} aria-label="Send question">
          {disabled
            ? <span className={styles.sendSpinner} />
            : <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
          }
        </button>
      </div>
    </div>
  )
}
