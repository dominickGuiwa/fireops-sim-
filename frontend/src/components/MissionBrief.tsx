import type { SimulationState } from '../types/simulation';
import {
  ArrowRightIcon,
  BrainIcon,
  EyeIcon,
  ShieldIcon,
  SparkIcon,
} from './Icons';
import { InfoTip } from './InfoTip';

interface MissionBriefProps {
  state: SimulationState;
}

export function MissionBrief({ state }: MissionBriefProps) {
  return (
    <section className="panel mission-brief" aria-labelledby="mission-title">
      <div className="mission-brief__goal">
        <span className="mission-number">MISSION 01</span>
        <div>
          <span className="section-kicker">Your goal</span>
          <h1 id="mission-title">Keep all 3 synthetic sites safe and help the fire go out.</h1>
          <p>
            The mission has two jobs: get the fire out before the step limit and keep all
            3 sites safe. Terrain and sites stay visible, but sensors may hide fire readings.
          </p>
        </div>
      </div>

      <div className="mission-live" aria-label="Live mission status">
        <div>
          <span>Sites safe</span>
          <strong>
            {state.metrics.infrastructurePreserved}/{state.metrics.infrastructureTotal || 3}
          </strong>
        </div>
        <div>
          <span>Fire now</span>
          <strong>{state.metrics.burningCells}</strong>
        </div>
        <div>
          <span>Resources left</span>
          <strong>{state.resources.available}</strong>
        </div>
        <div>
          <span>Step</span>
          <strong>
            {state.stepCount}
            {state.maxSteps ? <small>/{state.maxSteps}</small> : null}
          </strong>
        </div>
      </div>

      <div className="mission-flow" aria-label="What happens in three steps">
        <div className="mission-flow__step">
          <span className="mission-flow__number">1</span>
          <EyeIcon />
          <div>
            <strong>See</strong>
            <p>The agent gets a map with possible gaps.</p>
          </div>
        </div>
        <ArrowRightIcon className="mission-flow__arrow" />
        <div className="mission-flow__step">
          <span className="mission-flow__number">2</span>
          <BrainIcon />
          <div>
            <strong>Choose</strong>
            <p>A fixed policy picks one allowed move.</p>
          </div>
        </div>
        <ArrowRightIcon className="mission-flow__arrow" />
        <div className="mission-flow__step">
          <span className="mission-flow__number">3</span>
          <SparkIcon />
          <div>
            <strong>Score</strong>
            <p>
              See what changed and how many penalty points the step cost.
              <InfoTip term="Reward">
                Reward is the simulation score. A better step loses fewer points. New
                spread, site loss, resource use, and active fire subtract points.
              </InfoTip>
            </p>
          </div>
        </div>
      </div>

      <div className="mission-brief__safety">
        <ShieldIcon />
        This is a synthetic learning game, not real emergency guidance.
      </div>
    </section>
  );
}
