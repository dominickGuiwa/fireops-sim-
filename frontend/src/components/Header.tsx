import { ActivityIcon, FlameIcon, HelpIcon, RefreshIcon } from './Icons';
import type { ConnectionStatus } from '../types/simulation';

interface HeaderProps {
  connection: ConnectionStatus;
  step: number;
  isRunning: boolean;
  onRefresh: () => void;
  onHelp: () => void;
}

const connectionCopy: Record<ConnectionStatus, string> = {
  connecting: 'Linking API',
  online: 'API online',
  offline: 'Demo feed',
};

export function Header({ connection, step, isRunning, onRefresh, onHelp }: HeaderProps) {
  return (
    <header className="command-header">
      <div className="brand-lockup">
        <div className="brand-mark">
          <FlameIcon />
          <span className="brand-mark__pulse" />
        </div>
        <div>
          <div className="brand-name">
            FIREOPS<span>—SIM</span>
          </div>
          <div className="brand-subtitle">AI Wildfire Learning Lab</div>
        </div>
      </div>

      <div className="header-mission">
        <span className="eyebrow">Active scenario</span>
        <strong>SYN-WF / GRID 07</strong>
        <span className="header-mission__divider" />
        <span className="eyebrow">Cycle</span>
        <strong className="tabular">T+{String(step).padStart(3, '0')}</strong>
      </div>

      <div className="header-status">
        <div className={'run-state ' + (isRunning ? 'run-state--active' : '')}>
          <ActivityIcon />
          <span>{isRunning ? 'Simulation running' : 'Simulation paused'}</span>
        </div>
        <button
          className="help-button"
          type="button"
          onClick={onHelp}
          aria-label="Help"
        >
          <HelpIcon />
          <span>Help</span>
        </button>
        <button
          className={'connection-badge connection-badge--' + connection}
          type="button"
          onClick={onRefresh}
          title="Retry API connection"
        >
          <span className="connection-badge__dot" />
          {connectionCopy[connection]}
          <RefreshIcon className="connection-badge__refresh" />
        </button>
      </div>
    </header>
  );
}
