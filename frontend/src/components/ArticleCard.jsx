import styles from './ArticleCard.module.css'

export default function ArticleCard({ articleNum, title, snippet, score }) {
  const scoreColor = score >= 0.7 ? '#10b981' : score >= 0.4 ? '#f59e0b' : '#6b7280'
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.badge}>Art. {articleNum}</span>
        <span className={styles.title}>{title || 'Constitutional Provision'}</span>
        {score !== undefined && (
          <span className={styles.score} style={{ color: scoreColor }}>
            {(score * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {snippet && <p className={styles.snippet}>"{snippet}"</p>}
    </div>
  )
}
