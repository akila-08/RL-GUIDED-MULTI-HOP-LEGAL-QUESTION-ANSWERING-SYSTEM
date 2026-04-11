import AgentBrainPanel from './AgentBrainPanel'
import ThinkingIndicator from './ThinkingIndicator'
import styles from './MessageBubble.module.css'

export default function MessageBubble({ role, content, metadata, isThinking }) {
  const isUser = role === 'user'

  return (
    <div className={`${styles.wrapper} ${isUser ? styles.wrapperUser : styles.wrapperAssistant}`}>
      {/* Avatar */}
      <div className={`${styles.avatar} ${isUser ? styles.avatarUser : styles.avatarBot}`}>
        {isUser ? '👤' : '⚖️'}
      </div>

      {/* Bubble */}
      <div className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleBot}`}>
        {isThinking
          ? <ThinkingIndicator />
          : <>
              <p className={styles.content}>{content}</p>
              {metadata && <AgentBrainPanel metadata={metadata} />}
            </>
        }
      </div>
    </div>
  )
}
