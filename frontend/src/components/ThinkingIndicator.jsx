import styles from './ThinkingIndicator.module.css'

export default function ThinkingIndicator() {
  return (
    <div className={styles.wrapper}>
      <span className={styles.dot} style={{ animationDelay: '0s' }} />
      <span className={styles.dot} style={{ animationDelay: '0.18s' }} />
      <span className={styles.dot} style={{ animationDelay: '0.36s' }} />
    </div>
  )
}
