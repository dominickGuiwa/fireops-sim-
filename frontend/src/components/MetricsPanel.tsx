import type { SimulationState } from '../types/simulation';
import { FlameIcon, ResourceIcon, ShieldIcon, TargetIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface MetricsPanelProps {
  state: SimulationState;
}

function formatReward(value: number): string {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1);
  return value > 0 ? '+' + rounded : rounded;
}

export function MetricsPanel({ state }: MetricsPanelProps) {
  const infrastructurePercent = Math.round(
    (state.metrics.infrastructurePreserved /
      Math.max(1, state.metrics.infrastructureTotal)) *
      100,
  );
  const burnedPercent = Math.round(
    (state.metrics.burnedArea / Math.max(1, state.gridSize ** 2)) * 100,
  );

  return (
    <section className="panel metrics-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">What the score means</span>
          <h2>SCOREBOARD</h2>
        </div>
        <span className="panel-code">STEP {String(state.stepCount).padStart(3, '0')}</span>
      </div>

      <div className="metric-grid">
        <article className="metric-card metric-card--reward">
          <div className="metric-card__icon"><TargetIcon /></div>
          <span className="metric-card__label">
            Total reward
            <InfoTip term="Total reward">
              A better step loses fewer points. New spread, site loss, resource use,
              and active fire subtract points.
            </InfoTip>
          </span>
          <strong className={state.metrics.cumulativeReward < 0 ? 'value-critical' : ''}>
            {formatReward(state.metrics.cumulativeReward)}
          </strong>
          <small>score across all completed steps</small>
        </article>
        <article className="metric-card">
          <div className="metric-card__icon"><ShieldIcon /></div>
          <span>Sites safe</span>
          <strong>
            {state.metrics.infrastructurePreserved}
            <em>/{state.metrics.infrastructureTotal}</em>
          </strong>
          <small>{infrastructurePercent}% preserved</small>
        </article>
        <article className="metric-card metric-card--fire">
          <div className="metric-card__icon"><FlameIcon /></div>
          <span>Fire-affected area</span>
          <strong>
            {state.metrics.burnedArea}
            <em> cells</em>
          </strong>
          <small>{burnedPercent}% of grid (burning + burned)</small>
        </article>
        <article className="metric-card">
          <div className="metric-card__icon"><ResourceIcon /></div>
          <span>Resources used</span>
          <strong>
            {state.metrics.resourcesConsumed}
            <em>/{state.resources.initial}</em>
          </strong>
          <small>{state.metrics.burningCells} active fire cells</small>
        </article>
      </div>
    </section>
  );
}
