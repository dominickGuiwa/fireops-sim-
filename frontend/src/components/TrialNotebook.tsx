import { useMemo, useState } from 'react';
import {
  computePracticeStats,
  serverSessionLabel,
  type PracticeLog,
} from '../lib/trialLog';
import { outcomeLabel } from '../lib/guidance';
import type { TrialStats } from '../types/simulation';
import { BookIcon, ResetIcon, SparkIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface TrialNotebookProps {
  log: PracticeLog;
  serverStats: TrialStats | null;
  onClear: () => void;
  onViewLatest: () => void;
  canViewLatest: boolean;
}

function formatReward(value: number | null): string {
  if (value === null) return '—';
  const formatted = value.toFixed(1);
  return value > 0 ? '+' + formatted : formatted;
}

export function TrialNotebook({
  log,
  serverStats,
  onClear,
  onViewLatest,
  canViewLatest,
}: TrialNotebookProps) {
  const [confirmingClear, setConfirmingClear] = useState(false);
  const stats = useMemo(() => computePracticeStats(log), [log]);
  const serverLabel = serverSessionLabel(serverStats);

  return (
    <section className="panel notebook" aria-labelledby="notebook-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">All practice trials — this browser</span>
          <h2 id="notebook-title">DEMO AGENT NOTEBOOK</h2>
        </div>
        <BookIcon className="notebook-heading-icon" />
      </div>

      <p className="notebook-xp-note">
        <SparkIcon />
        <span>
          XP tracks practice in this browser. It does not train the agent.
          <InfoTip term="XP">
            XP is a score for your completed practice trials. It does not change any
            policy or retrain a model.
          </InfoTip>
        </span>
      </p>

      <div className="notebook-stats">
        <div>
          <span>All-time trials</span>
          <strong>{stats.allTimeTrials}</strong>
        </div>
        <div>
          <span>Success rate</span>
          <strong>{Math.round(stats.successRate * 100)}%</strong>
        </div>
        <div>
          <span>Average / best</span>
          <strong>{formatReward(stats.averageReward)} <em>/ {formatReward(stats.bestReward)}</em></strong>
        </div>
        <div>
          <span>XP / level</span>
          <strong>{stats.totalXp} <em>/ L{stats.level}</em></strong>
        </div>
      </div>

      <div className="notebook-lesson">
        <span>Recurring lesson</span>
        <strong>
          {stats.recurringLesson ?? 'Finish a trial to reveal a practice lesson.'}
        </strong>
      </div>

      <div className="recent-trials">
        <div className="recent-trials__heading">
          <span>Recent outcomes saved here</span>
          {log.recent.length > 0 && canViewLatest ? (
            <button type="button" onClick={onViewLatest}>View latest report</button>
          ) : null}
        </div>
        {log.recent.length === 0 ? (
          <p className="notebook-empty">No completed practice trials yet.</p>
        ) : (
          <div className="recent-trials__list">
            {log.recent.slice(0, 5).map((trial, index) => (
              <article key={trial.id}>
                <i className={'outcome-dot outcome-dot--' + trial.outcome} />
                <div>
                  <span>
                    {index === 0
                      ? 'Most recent'
                      : index === 1
                        ? '1 run ago'
                        : index + ' runs ago'}
                  </span>
                  <strong>{outcomeLabel(trial.outcome)}</strong>
                </div>
                <span className="recent-trial__reward">{formatReward(trial.reward)}</span>
              </article>
            ))}
          </div>
        )}
      </div>

      {serverLabel ? <p className="server-session-note">{serverLabel}</p> : null}

      {confirmingClear ? (
        <div className="clear-confirm" role="alert">
          <span>Clear all saved practice data in this browser?</span>
          <button type="button" onClick={() => setConfirmingClear(false)}>Cancel</button>
          <button
            className="clear-confirm__danger"
            type="button"
            onClick={() => {
              onClear();
              setConfirmingClear(false);
            }}
          >
            Clear data
          </button>
        </div>
      ) : (
        <button
          className="clear-practice-button"
          type="button"
          onClick={() => setConfirmingClear(true)}
        >
          <ResetIcon />
          Clear practice data
        </button>
      )}
    </section>
  );
}
