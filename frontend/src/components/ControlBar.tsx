import { useState } from 'react';
import { ACTIONS } from '../types/simulation';
import {
  ChaosIcon,
  ChevronIcon,
  PauseIcon,
  PlayIcon,
  ResetIcon,
  StepIcon,
} from './Icons';

interface ControlBarProps {
  isRunning: boolean;
  isBusy: boolean;
  done: boolean;
  stepCount: number;
  onRun: () => void;
  onPause: () => void;
  onStep: (action?: string) => void;
  onReset: () => void;
  onChaos: () => void;
}

export function ControlBar({
  isRunning,
  isBusy,
  done,
  stepCount,
  onRun,
  onPause,
  onStep,
  onReset,
  onChaos,
}: ControlBarProps) {
  const [manualAction, setManualAction] = useState('');

  return (
    <footer className="control-deck">
      <div className="control-deck__status">
        <span className={'control-state-light ' + (isRunning ? 'is-active' : '')} />
        <div>
          <span>{done ? 'Trial complete' : isRunning ? 'Simulation running' : 'Simulation paused'}</span>
          <strong>CYCLE {String(stepCount).padStart(3, '0')}</strong>
        </div>
      </div>

      <div className="manual-action">
        <label htmlFor="manual-action">Next-step move</label>
        <div className="select-shell select-shell--compact">
          <select
            id="manual-action"
            value={manualAction}
            onChange={(event) => setManualAction(event.target.value)}
          >
            <option value="">Fixed policy chooses</option>
            {ACTIONS.map((action) => (
              <option value={action.value} key={action.value}>
                {String(action.id).padStart(2, '0')} · {action.label}
              </option>
            ))}
          </select>
          <ChevronIcon />
        </div>
      </div>

      <div className="control-buttons">
        <button
          className="control-button control-button--secondary"
          type="button"
          onClick={onReset}
          disabled={isBusy}
        >
          <ResetIcon />
          <span>Reset</span>
        </button>
        <button
          className={'control-button control-button--run ' + (isRunning ? 'is-running' : '')}
          type="button"
          onClick={isRunning ? onPause : onRun}
          disabled={isBusy || done}
        >
          {isRunning ? <PauseIcon /> : <PlayIcon />}
          <span>{isRunning ? 'Pause' : 'Run'}</span>
        </button>
        <button
          className="control-button control-button--secondary"
          type="button"
          onClick={() => onStep(manualAction || undefined)}
          disabled={isBusy || isRunning || done}
        >
          <StepIcon />
          <span>Step</span>
        </button>
        <span className="control-divider" />
        <button
          className="control-button control-button--chaos"
          type="button"
          onClick={onChaos}
          disabled={isBusy || done}
        >
          <ChaosIcon />
          <span>Add chaos</span>
        </button>
      </div>
      {isBusy ? <span className="command-progress" aria-label="Command in progress" /> : null}
    </footer>
  );
}
