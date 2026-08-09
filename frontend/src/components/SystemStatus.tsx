import type { SimulationState } from '../types/simulation';
import { RadioIcon, ResourceIcon, ShieldIcon } from './Icons';

interface SystemStatusProps {
  state: SimulationState;
}

function statusTone(value: 'good' | 'warning' | 'critical'): string {
  return 'status-row status-row--' + value;
}

export function SystemStatus({ state }: SystemStatusProps) {
  const observed = Math.round(state.sensor.observedFraction * 100);
  const reliability = Math.round(state.sensor.reliability * 100);
  const resourcePercent = Math.round(
    (state.resources.available / Math.max(1, state.resources.initial)) * 100,
  );
  const sensorTone =
    state.sensor.status === 'online'
      ? 'good'
      : state.sensor.status === 'degraded'
        ? 'warning'
        : 'critical';
  const commsTone = state.sensor.communicationDegraded ? 'critical' : 'good';
  const resourceTone =
    resourcePercent > 50 ? 'good' : resourcePercent > 20 ? 'warning' : 'critical';

  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Visible information only</span>
          <h2>WHAT THE AGENT KNOWS</h2>
        </div>
        <span className="live-stamp">
          <i />
          LIVE
        </span>
      </div>

      <div className={statusTone(sensorTone)}>
        <div className="status-row__icon">
          <RadioIcon />
        </div>
        <div className="status-row__body">
          <div className="status-row__heading">
            <strong>Fire readings</strong>
            <span>{state.sensor.status}</span>
          </div>
          <div className="status-bar">
            <i style={{ width: reliability + '%' }} />
          </div>
          <small>{observed}% of cells have fire data · {reliability}% effective reliability</small>
        </div>
      </div>

      <div className={statusTone(commsTone)}>
        <div className="status-row__icon">
          <ShieldIcon />
        </div>
        <div className="status-row__body">
          <div className="status-row__heading">
            <strong>Map data link</strong>
            <span>{state.sensor.communicationDegraded ? 'degraded' : 'nominal'}</span>
          </div>
          <div className="signal-bars" aria-hidden="true">
            {[1, 2, 3, 4, 5].map((bar) => (
              <i
                key={bar}
                className={
                  state.sensor.communicationDegraded && bar > 2 ? 'signal-bars__inactive' : ''
                }
              />
            ))}
          </div>
          <small>
            {state.sensor.communicationDegraded
                ? 'A synthetic data-link fault hides more information'
                : 'The agent is receiving its simulated map normally'}
          </small>
        </div>
      </div>

      <div className={statusTone(resourceTone)}>
        <div className="status-row__icon">
          <ResourceIcon />
        </div>
        <div className="status-row__body">
          <div className="status-row__heading">
            <strong>Resources left</strong>
            <span>{state.resources.available} / {state.resources.initial}</span>
          </div>
          <div className="resource-pips" aria-label={state.resources.available + ' resources available'}>
            {Array.from({ length: state.resources.initial }, (_, index) => (
              <i
                key={index}
                className={
                  index < state.resources.available
                    ? 'is-available'
                    : index < state.resources.available + state.resources.unavailable
                      ? 'is-unavailable'
                      : ''
                }
              />
            ))}
          </div>
          <small>
            {state.resources.consumed} consumed
            {state.resources.unavailable > 0 ? ' · ' + state.resources.unavailable + ' unavailable' : ' · no resource faults'}
          </small>
        </div>
      </div>
    </section>
  );
}
