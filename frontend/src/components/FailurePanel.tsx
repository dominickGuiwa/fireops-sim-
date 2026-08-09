import type { AgentFailure, AgentFailureType, SimulationState } from '../types/simulation';
import { AlertIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface FailurePanelProps {
  state: SimulationState;
}

const FAILURE_LABELS: Record<AgentFailureType, string> = {
  perception_gap: 'Unseen fire',
  misdirected_protection: 'Wrong sector',
  idle_while_burning: 'Idle in fire',
  resource_exhausted_request: 'No resources',
  wind_misread: 'Wind misread',
};

function failureCounts(failures: AgentFailure[]): Array<[AgentFailureType, number]> {
  const counts = new Map<AgentFailureType, number>();
  for (const failure of failures) {
    counts.set(failure.failureType, (counts.get(failure.failureType) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

export function FailurePanel({ state }: FailurePanelProps) {
  const failures = state.agentFailures;
  const counts = failureCounts(failures);
  const latest = failures.slice(-4).reverse();

  return (
    <section className="panel failure-panel" aria-labelledby="failure-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">When the agent goes wrong</span>
          <h2 id="failure-title">AGENTIC FAILURE ANALYSIS</h2>
        </div>
        <span className={'failure-count' + (failures.length > 0 ? ' failure-count--active' : '')}>
          {failures.length}
          <InfoTip term="Failure analysis">
            Each flagged step compares the move the agent chose from its degraded
            map against the true synthetic world. The truth is used only for this
            analysis—never given to the agent.
          </InfoTip>
        </span>
      </div>

      {failures.length === 0 ? (
        <p className="failure-empty">
          No agentic failures flagged this trial yet. Lower map reliability or
          inject chaos, then watch how degraded information leads the agent into
          bad moves.
        </p>
      ) : (
        <>
          <div className="failure-tags" role="list">
            {counts.map(([type, count]) => (
              <span className="failure-tag" role="listitem" key={type}>
                {FAILURE_LABELS[type]}
                <strong>{count}</strong>
              </span>
            ))}
          </div>
          <ul className="failure-list">
            {latest.map((failure) => (
              <li key={failure.id}>
                <span className="failure-step">
                  <AlertIcon />T+{failure.step}
                </span>
                <p>{failure.message}</p>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
