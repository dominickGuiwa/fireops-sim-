import type { PolicyDefinition, SimulationState } from '../types/simulation';
import { actionLabel } from '../lib/simulation';
import { ActivityIcon, ChevronIcon, TargetIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface PolicyPanelProps {
  state: SimulationState;
  policies: PolicyDefinition[];
  onSelectPolicy: (policy: string) => void;
  onReliabilityChange: (value: number) => void;
  onAddAi: () => void;
}

export function PolicyPanel({
  state,
  policies,
  onSelectPolicy,
  onReliabilityChange,
  onAddAi,
}: PolicyPanelProps) {
  const selectedPolicy =
    policies.find((policy) => policy.id === state.policy) ?? policies[0];
  const reliability = Math.round(state.sensor.configuredReliability * 100);

  return (
    <section className="panel policy-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Choose who plays</span>
          <h2>AI AGENT</h2>
        </div>
        <span className="panel-code">CTRL—02</span>
      </div>

      <div className="field-group">
        <div className="field-label-row">
          <label htmlFor="policy-select">Decision policy</label>
          <InfoTip term="Decision policy">
            A policy is the fixed set of rules the agent uses to choose a move from
            the information it can see.
          </InfoTip>
        </div>
        <div className="select-shell">
          <select
            id="policy-select"
            value={state.policy}
            onChange={(event) => onSelectPolicy(event.target.value)}
          >
            {policies.map((policy) => (
              <option value={policy.id} key={policy.id}>
                {policy.label}
              </option>
            ))}
          </select>
          <ChevronIcon />
        </div>
        {selectedPolicy ? (
          <div className="policy-description">
            <span className={'policy-category policy-category--' + selectedPolicy.category}>
              {selectedPolicy.category}
            </span>
            <p>{selectedPolicy.description}</p>
            {!selectedPolicy.nativeAvailable ? (
              <p className="policy-fallback-note">
                PPO is unavailable in this build, so the server uses the Rule Agent
                fallback.
              </p>
            ) : null}
          </div>
        ) : null}
        <button type="button" className="add-ai-button" onClick={onAddAi}>
          + Add AI <span>prototype</span>
        </button>
      </div>

      <div className="reliability-control">
        <div className="reliability-control__heading">
          <div>
            <div className="field-label-row">
              <label htmlFor="reliability">Map reliability</label>
              <InfoTip term="Map reliability">
                Lower reliability removes more fire readings from Agent View. Terrain
                and sites remain visible. The Truth view is not shown to the agent.
              </InfoTip>
            </div>
            <small>Changing this pauses the simulation and starts a new trial</small>
          </div>
          <strong className={reliability < 50 ? 'value-critical' : ''}>{reliability}%</strong>
        </div>
        <input
          id="reliability"
          type="range"
          min="25"
          max="100"
          step="1"
          value={reliability}
          onChange={(event) => onReliabilityChange(Number(event.target.value) / 100)}
          style={{ '--range-value': String((reliability - 25) / 75) } as React.CSSProperties}
        />
        <div className="range-ticks" aria-hidden="true">
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
      </div>

      <div className="current-action">
        <div className="current-action__icon">
          <TargetIcon />
          <span />
        </div>
        <div className="current-action__copy">
          <span>Move chosen</span>
          <strong>{actionLabel(state.currentAction)}</strong>
          <small>
            {state.sensor.reconActive
              ? 'Recon briefly reveals more map cells'
              : 'Chosen by ' + (selectedPolicy?.label ?? state.policy)}
          </small>
        </div>
        <ActivityIcon className="current-action__activity" />
      </div>
    </section>
  );
}
