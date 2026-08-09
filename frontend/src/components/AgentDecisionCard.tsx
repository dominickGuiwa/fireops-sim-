import { useEffect, useMemo, useRef } from 'react';
import { explainDecision } from '../lib/guidance';
import type { PolicyDefinition, SimulationState } from '../types/simulation';
import { BrainIcon, EyeIcon, SparkIcon, TargetIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface AgentDecisionCardProps {
  state: SimulationState;
  policy: PolicyDefinition | undefined;
}

function signed(value: number | null): string {
  if (value === null) return '—';
  const rounded = Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return value > 0 ? '+' + rounded : rounded;
}

export function AgentDecisionCard({ state, policy }: AgentDecisionCardProps) {
  const previousReward = useRef<number | null>(null);
  const previousStep = useRef<number | null>(null);
  const comparableReward =
    previousStep.current !== null && state.stepCount === previousStep.current + 1
      ? previousReward.current
      : null;
  const explanation = useMemo(
    () => explainDecision(state, comparableReward, policy),
    [comparableReward, policy, state],
  );

  useEffect(() => {
    previousReward.current = state.metrics.cumulativeReward;
    previousStep.current = state.stepCount;
  }, [state.metrics.cumulativeReward, state.stepCount]);

  return (
    <section className="panel decision-card" aria-labelledby="decision-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">One step, explained</span>
          <h2 id="decision-title">AGENT DECISION — EXPLAINED</h2>
        </div>
        <span className="decision-source">{explanation.sourceLabel}</span>
      </div>

      {explanation.fallbackNotice ? (
        <div className="fallback-notice" role="note">
          <strong>Rule fallback in use</strong>
          {explanation.fallbackNotice}
        </div>
      ) : null}

      <div className="decision-loop">
        <article>
          <span className="decision-loop__icon"><EyeIcon /></span>
          <div>
            <span>What it can see</span>
            <p>{explanation.seen}</p>
          </div>
        </article>
        <article>
          <span className="decision-loop__icon"><TargetIcon /></span>
          <div>
            <span>Move picked</span>
            <strong>{explanation.picked}</strong>
          </div>
        </article>
        <article>
          <span className="decision-loop__icon"><BrainIcon /></span>
          <div>
            <span>
              Why
              <InfoTip term="Why">
                A reason sent by the server is shown as-is. If it is missing, we show a
                simple hint and say that it is inferred.
              </InfoTip>
            </span>
            <p>{explanation.why}</p>
          </div>
        </article>
      </div>

      <div className="decision-reward">
        <div>
          <span>
            This step
            <InfoTip term="Step reward">
              The score change from only the newest move. Penalties make it zero or
              lower.
            </InfoTip>
          </span>
          <strong
            className={
              explanation.stepReward !== null && explanation.stepReward < 0
                ? 'value-critical'
                : ''
            }
          >
            {signed(explanation.stepReward)}
          </strong>
        </div>
        <SparkIcon />
        <div>
          <span>
            Total reward
            <InfoTip term="Total reward">
              The sum of every step score in this trial. A better run loses fewer
              points.
            </InfoTip>
          </span>
          <strong>{signed(explanation.totalReward)}</strong>
        </div>
      </div>

      {state.agentDecision?.rewardBreakdown ? (
        <details className="reward-breakdown">
          <summary>Show step score parts</summary>
          <div>
            <span>New fire spread <strong>{signed(state.agentDecision.rewardBreakdown.newSpreadPenalty)}</strong></span>
            <span>Site damage <strong>{signed(state.agentDecision.rewardBreakdown.infrastructureLossPenalty)}</strong></span>
            <span>Resource use <strong>{signed(state.agentDecision.rewardBreakdown.resourceUsePenalty)}</strong></span>
            <span>Active fire <strong>{signed(state.agentDecision.rewardBreakdown.activeFirePenalty)}</strong></span>
          </div>
        </details>
      ) : null}

      {!explanation.hasDecision ? (
        <p className="decision-caveat">
          The agent will use only Agent View when the first step begins.
        </p>
      ) : explanation.isInferred ? (
        <p className="decision-caveat">
          The API did not provide a reason. This is a plain-language hint from the visible
          state and action—not the policy&apos;s hidden thoughts.
        </p>
      ) : (
        <p className="decision-caveat decision-caveat--server">
          This explanation came from the simulator&apos;s decision record.
        </p>
      )}
    </section>
  );
}
