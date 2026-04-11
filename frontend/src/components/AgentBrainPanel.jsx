import { useState } from 'react'
import ArticleCard from './ArticleCard'
import styles from './AgentBrainPanel.module.css'

const SectionHeader = ({ icon, label, count }) => (
  <div className={styles.sectionHeader}>
    <span className={styles.sectionIcon}>{icon}</span>
    <span className={styles.sectionLabel}>{label}</span>
    {count !== undefined && <span className={styles.sectionCount}>{count}</span>}
  </div>
)

export default function AgentBrainPanel({ metadata }) {
  const [open, setOpen] = useState(false)
  if (!metadata) return null

  const { is_complex, complexity_score, actions_taken = [], sub_questions = [],
          retrieved_articles = [], sub_answers = [], rewards = {}, combined_reward } = metadata

  const rewardPct = Math.min(100, Math.max(0, (combined_reward || 0) * 100))
  const rewardColor = rewardPct >= 60 ? '#10b981' : rewardPct >= 35 ? '#f59e0b' : '#ef4444'

  return (
    <div className={styles.wrapper}>
      {/* Badge row */}
      <div className={styles.badgeRow}>
        <span className={is_complex ? styles.badgeComplex : styles.badgeSimple}>
          {is_complex ? '🔀 Multi-Hop HRL' : '⚡ Single-Hop'}
        </span>
        <span className={styles.badgePipeline}>{actions_taken.join(' → ')}</span>
        <button className={styles.toggleBtn} onClick={() => setOpen(!open)}>
          {open ? '▲ Hide' : '▼ Agent Brain'} 
        </button>
      </div>

      {/* Collapsible panel */}
      {open && (
        <div className={styles.panel}>
          {/* Complexity bar */}
          <div className={styles.metricRow}>
            <span className={styles.metricLabel}>Complexity</span>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{
                width: `${(complexity_score || 0) * 100}%`,
                background: is_complex
                  ? 'linear-gradient(90deg, #7c3aed, #a78bfa)'
                  : 'linear-gradient(90deg, #10b981, #6ee7b7)',
              }} />
            </div>
            <span className={styles.metricVal}>{((complexity_score || 0) * 100).toFixed(0)}%</span>
          </div>

          {/* Sub questions */}
          {sub_questions.length > 0 && (
            <div className={styles.section}>
              <SectionHeader icon="📝" label="Decomposed Questions" count={sub_questions.length} />
              <ol className={styles.subQList}>
                {sub_questions.map((q, i) => (
                  <li key={i} className={styles.subQItem}>{q}</li>
                ))}
              </ol>
            </div>
          )}

          {/* Retrieved articles */}
          {retrieved_articles.length > 0 && (
            <div className={styles.section}>
              <SectionHeader icon="📜" label="Retrieved Articles" count={retrieved_articles.length} />
              <div className={styles.articleGrid}>
                {retrieved_articles.map((a, i) => (
                  <ArticleCard
                    key={i}
                    articleNum={a.article_num}
                    title={a.title}
                    snippet={a.text_snippet}
                    score={a.rerank_score}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Sub answers */}
          {sub_answers.length > 0 && (
            <div className={styles.section}>
              <SectionHeader icon="💡" label="Sub-Answers" count={sub_answers.length} />
              {sub_answers.map((sa, i) => (
                <div key={i} className={styles.subAnswer}>{sa}</div>
              ))}
            </div>
          )}

          {/* Reward */}
          <div className={styles.section}>
            <SectionHeader icon="🏆" label="RL Reward" />
            <div className={styles.rewardRow}>
              <div className={styles.rewardGauge}>
                <svg viewBox="0 0 80 50" className={styles.gaugeSvg}>
                  {/* Track */}
                  <path d="M10,45 A30,30 0 0,1 70,45" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="6" strokeLinecap="round" />
                  {/* Fill */}
                  <path d="M10,45 A30,30 0 0,1 70,45" fill="none" stroke={rewardColor}
                    strokeWidth="6" strokeLinecap="round"
                    strokeDasharray={`${rewardPct * 0.942} 100`}
                    style={{ filter: `drop-shadow(0 0 4px ${rewardColor})` }}
                  />
                  <text x="40" y="44" textAnchor="middle" fontSize="11" fontWeight="700" fill={rewardColor}>
                    {(combined_reward || 0).toFixed(3)}
                  </text>
                </svg>
              </div>
              <div className={styles.rewardBreakdown}>
                {Object.entries(rewards).map(([k, v]) => (
                  <div key={k} className={styles.rewardItem}>
                    <span className={styles.rewardKey}>{k}</span>
                    <span className={styles.rewardVal} style={{ color: v > 0.5 ? '#10b981' : '#f59e0b' }}>
                      {v.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
