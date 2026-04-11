import React from 'react'
import styles from './ThinkingIndicator.module.css'

const steps = ['DECOMPOSE', 'RETRIEVE', 'GENERATE', 'COMBINE']

export default function ThinkingIndicator({ currentStep = 0 }) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <span className={styles.spinner} />
        <span className={styles.label}>RL Agent is thinking…</span>
      </div>
      <div className={styles.pipeline}>
        {steps.map((step, i) => (
          <React.Fragment key={step}>
            <div className={`${styles.step} ${i <= currentStep ? styles.active : ''} ${i === currentStep ? styles.current : ''}`}>
              {step}
            </div>
            {i < steps.length - 1 && <div className={`${styles.arrow} ${i < currentStep ? styles.arrowDone : ''}`}>→</div>}
          </React.Fragment>
        ))}
      </div>
      <div className={styles.dots}>
        {[0,1,2].map(i => <span key={i} className={styles.dot} style={{ animationDelay: `${i * 0.18}s` }} />)}
      </div>
    </div>
  )
}

