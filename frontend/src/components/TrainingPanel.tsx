import { useMemo, useState } from 'react';
import type {
  ConnectionStatus,
  LearningEpisode,
  SimulationState,
  TrainingDifficulty,
} from '../types/simulation';
import { BrainIcon, ChevronIcon, SparkIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface TrainingPanelProps {
  state: SimulationState;
  connection: ConnectionStatus;
  isTraining: boolean;
  isBusy: boolean;
  onTrain: (episodes: number, difficulty: TrainingDifficulty) => void;
}

const TRAIN_BATCH = 25;
const SPARKLINE_POINTS = 60;

const DIFFICULTY_COPY: Record<TrainingDifficulty, string> = {
  standard: 'Current settings, unchanged worlds.',
  hard: 'Ember jumps, wind drift, sensor dropout; reliability 50-100% per episode.',
  brutal:
    'Stronger chaos plus one step of stale sensor data; reliability 25-90% per episode.',
};

function signed(value: number | null): string {
  if (value === null) return '—';
  const rounded = Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return value > 0 ? '+' + rounded : rounded;
}

function Sparkline({ history }: { history: LearningEpisode[] }) {
  const points = history.slice(-SPARKLINE_POINTS);
  const path = useMemo(() => {
    if (points.length < 2) return null;
    const rewards = points.map((item) => item.reward);
    const min = Math.min(...rewards);
    const max = Math.max(...rewards);
    const span = max - min || 1;
    return points
      .map((item, index) => {
        const x = (index / (points.length - 1)) * 100;
        const y = 30 - ((item.reward - min) / span) * 26 - 2;
        return (index === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
      })
      .join(' ');
  }, [points]);

  if (!path) {
    return (
      <p className="training-empty">
        No completed episodes yet. Train the agent or finish a trial with the
        Learning policy selected.
      </p>
    );
  }
  return (
    <svg
      className="training-sparkline"
      viewBox="0 0 100 32"
      preserveAspectRatio="none"
      role="img"
      aria-label={'Reward per completed learning episode, latest ' + points.length + ' episodes'}
    >
      <path d={path} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function TrainingPanel({
  state,
  connection,
  isTraining,
  isBusy,
  onTrain,
}: TrainingPanelProps) {
  const [difficulty, setDifficulty] = useState<TrainingDifficulty>('standard');
  const learning = state.learning;
  const offline = connection !== 'online';
  const improvement = learning?.improvement ?? null;
  const trend =
    improvement === null ? 'flat' : improvement > 0 ? 'up' : improvement < 0 ? 'down' : 'flat';

  return (
    <section className="panel training-panel" aria-labelledby="training-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Watch it learn</span>
          <h2 id="training-title">TRAINING PROGRESS</h2>
        </div>
        <span className="panel-code">CTRL—03</span>
      </div>

      {offline || !learning ? (
        <p className="training-empty">
          {offline
            ? 'Training runs on the backend. Reconnect the API link to train the learning policy and see real progress.'
            : 'Learning telemetry has not arrived yet.'}
        </p>
      ) : (
        <>
          <div className="training-stats">
            <div>
              <span>
                Episodes trained
                <InfoTip term="Episodes trained">
                  Completed practice runs the learning policy has updated its value
                  table from, counting both interactive trials and server-side
                  training batches in this server session.
                </InfoTip>
              </span>
              <strong>{learning.episodesTrained}</strong>
            </div>
            <div>
              <span>
                Exploration
                <InfoTip term="Exploration">
                  The chance the agent tries a random move instead of its current
                  best one. It shrinks as training goes on, so the agent settles on
                  what worked.
                </InfoTip>
              </span>
              <strong>{Math.round(learning.epsilon * 100)}%</strong>
            </div>
            <div>
              <span>
                Trend
                <InfoTip term="Trend">
                  Recent average episode reward minus the early average. Above zero
                  means recent episodes score better than the first ones.
                </InfoTip>
              </span>
              <strong className={'training-trend training-trend--' + trend}>
                {signed(improvement)}
              </strong>
            </div>
          </div>

          <Sparkline history={learning.rewardHistory} />
          {learning.earlyAverage !== null && learning.recentAverage !== null ? (
            <p className="training-compare">
              Early episodes averaged <strong>{signed(learning.earlyAverage)}</strong>; recent
              episodes average <strong>{signed(learning.recentAverage)}</strong>.
            </p>
          ) : null}

          <div className="training-difficulty">
            <label htmlFor="training-difficulty">
              Difficulty
              <InfoTip term="Training difficulty">
                Harder presets randomize each training world: ember jumps, wind
                drift, sensor dropout, stale observations, and lower sensor
                reliability. They exist to make training genuinely difficult.
              </InfoTip>
            </label>
            <div className="select-shell">
              <select
                id="training-difficulty"
                value={difficulty}
                onChange={(event) =>
                  setDifficulty(event.target.value as TrainingDifficulty)
                }
              >
                <option value="standard">Standard</option>
                <option value="hard">Hard</option>
                <option value="brutal">Brutal</option>
              </select>
              <ChevronIcon />
            </div>
            <small>{DIFFICULTY_COPY[difficulty]}</small>
          </div>

          <button
            type="button"
            className="training-button"
            disabled={isTraining || isBusy}
            onClick={() => onTrain(TRAIN_BATCH, difficulty)}
          >
            <BrainIcon />
            {isTraining
              ? 'Training…'
              : 'Train ' + TRAIN_BATCH + ' ' + difficulty + ' episodes'}
          </button>
          <p className="training-caption">
            <SparkIcon />
            Real tabular Q-learning on the server: fresh seeded synthetic worlds,
            value updates from rewards only. Progress lasts for this server session.
          </p>
        </>
      )}
    </section>
  );
}
