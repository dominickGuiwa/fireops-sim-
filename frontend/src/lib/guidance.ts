import { actionLabel } from './simulation';
import type {
  PolicyDefinition,
  SimulationState,
  TrialOutcome,
} from '../types/simulation';

export interface DecisionExplanation {
  seen: string;
  picked: string;
  why: string;
  stepReward: number | null;
  totalReward: number;
  confidence: number | null;
  sourceLabel: string;
  isInferred: boolean;
  hasDecision: boolean;
  fallbackNotice: string | null;
}

export interface GuidedRunReport {
  id: string;
  outcome: TrialOutcome;
  missionCompleted: boolean;
  successful: boolean;
  title: string;
  whatHappened: string[];
  rewardLesson: string;
  improvements: string[];
  xpEarned: number;
  terminationReason: string;
  totalReward: number;
  stepReward: number | null;
  infrastructurePreserved: number;
  infrastructureTotal: number;
  burnedArea: number;
  totalCells: number;
  resourcesConsumed: number;
  lessonCode: string;
  lesson: string;
  source: 'server summary' | 'result-based summary';
}

function formatCount(value: number, noun: string): string {
  return value + ' ' + noun + (value === 1 ? '' : 's');
}

function fallbackReason(action: string, policy: string): string {
  if (policy.startsWith('ppo_')) {
    return 'The server did not send a reason. We cannot tell which details the PPO policy weighed.';
  }
  if (policy === 'random') {
    return 'The Random agent chooses from the allowed moves without using a strategy.';
  }
  if (action === 'RECON') {
    return 'Some map squares were hidden, so the rule chose to look for more information.';
  }
  if (action.startsWith('PROTECT_')) {
    const side = action.replace('PROTECT_', '').toLowerCase();
    return 'The rule chose the ' + side + ' side to place protection near the fire risk.';
  }
  return 'The rule did not find a protection move that clearly beat waiting this step.';
}

export function explainDecision(
  state: SimulationState,
  previousTotalReward: number | null,
  policy: PolicyDefinition | undefined,
): DecisionExplanation {
  const decision = state.agentDecision;
  const fallbackNotice =
    policy && !policy.nativeAvailable
      ? policy.label +
        ' is not installed in this demo, so the server uses the Rule Agent fallback.'
      : null;
  const visibleCells =
    decision?.visibleCells ??
    (decision?.hiddenCells === null || decision?.hiddenCells === undefined
      ? state.cells.filter((cell) => cell.observed).length
      : Math.max(0, state.cells.length - decision.hiddenCells));
  const visibleFire =
    decision?.visibleFireCells ??
    state.cells.filter((cell) => cell.observed && cell.fire === 'burning').length;
  const hiddenCells =
    decision?.hiddenCells ?? Math.max(0, state.cells.length - visibleCells);
  const seen =
    decision?.seenSummary ??
    'It has fire readings for ' +
      formatCount(visibleCells, 'map square') +
      ', including ' +
      formatCount(visibleFire, 'fire square') +
      '. ' +
      formatCount(hiddenCells, 'fire reading') +
      ' are unavailable. Terrain and sites stay visible.';
  if (state.stepCount === 0 && !decision) {
    return {
      seen,
      picked: 'Waiting for the first move',
      why: 'Press Run or Step to let the fixed policy choose its first move.',
      stepReward: null,
      totalReward: state.metrics.cumulativeReward,
      confidence: null,
      sourceLabel: 'Ready',
      isInferred: false,
      hasDecision: false,
      fallbackNotice,
    };
  }
  const selectedAction = decision?.action ?? state.currentAction;
  const picked = decision?.actionLabel ?? actionLabel(selectedAction);
  const rewardFromTotals =
    previousTotalReward === null
      ? null
      : state.metrics.cumulativeReward - previousTotalReward;
  const stepReward = decision?.rewardDelta ?? state.lastReward ?? rewardFromTotals;
  const isInferred = !decision?.reason;
  const why = decision?.reason ?? fallbackReason(selectedAction, state.policy);
  return {
    seen,
    picked,
    why,
    stepReward,
    totalReward: state.metrics.cumulativeReward,
    confidence: decision?.confidence ?? null,
    sourceLabel:
      decision?.decisionSource === 'manual'
        ? 'You picked this one step'
        : 'Fixed policy choosing actions',
    isInferred,
    hasDecision: true,
    fallbackNotice,
  };
}

function outcomeFromState(state: SimulationState): TrialOutcome {
  if (state.runSummary?.outcome) return state.runSummary.outcome;
  const terminationReason =
    state.runSummary?.terminationReason ??
    state.terminationReason ??
    (state.metrics.burningCells === 0 ? 'fire_extinguished' : 'max_steps');
  if (terminationReason === 'fire_extinguished') {
    const infrastructureLost =
      state.runSummary?.finalMetrics?.infrastructureLost ??
      state.metrics.infrastructureLost;
    return infrastructureLost === 0 ? 'success' : 'partial';
  }
  return 'needs_improvement';
}

function outcomeTitle(outcome: TrialOutcome): string {
  if (outcome === 'success') return 'Mission success!';
  if (outcome === 'partial') return 'The fire went out, but a site was lost.';
  return 'This run needs another try.';
}

function fallbackImprovements(state: SimulationState, outcome: TrialOutcome): string[] {
  const suggestions: string[] = [];
  if (state.metrics.infrastructureLost > 0) {
    suggestions.push('Protect a threatened site earlier in the run.');
  }
  if (state.sensor.configuredReliability < 0.55) {
    suggestions.push('Try Recon when many map squares are hidden.');
  }
  if (state.resources.available === 0) {
    suggestions.push('Save one response resource for a later danger.');
  }
  if (outcome === 'needs_improvement' && suggestions.length === 0) {
    suggestions.push('Compare Agent View with Truth, then try a different fixed policy.');
  }
  if (suggestions.length === 0) {
    suggestions.push('Lower sensor reliability next time to test a harder case.');
  }
  return suggestions.slice(0, 2);
}

function recurringLesson(
  state: SimulationState,
  outcome: TrialOutcome,
): { code: string; text: string } {
  if (state.metrics.infrastructureLost > 0) {
    return { code: 'protect_sites', text: 'Protect threatened sites earlier.' };
  }
  if (state.sensor.configuredReliability < 0.55) {
    return { code: 'use_recon', text: 'Use Recon when the map has many gaps.' };
  }
  if (state.resources.available === 0) {
    return { code: 'save_resources', text: 'Keep a resource ready for later steps.' };
  }
  if (outcome === 'needs_improvement') {
    return { code: 'change_policy', text: 'Compare another fixed policy on the same challenge.' };
  }
  return { code: 'raise_difficulty', text: 'Try a harder sensor setting and compare the result.' };
}

function xpForRun(state: SimulationState, outcome: TrialOutcome): number {
  const outcomePoints = outcome === 'success' ? 60 : outcome === 'partial' ? 35 : 15;
  const sitePoints = state.metrics.infrastructurePreserved * 12;
  return Math.max(
    10,
    Math.min(200, Math.round(outcomePoints + sitePoints + Math.max(0, state.metrics.cumulativeReward))),
  );
}

function privacySafeHash(value: string): string {
  let hash = 2_166_136_261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }
  return (hash >>> 0).toString(36).padStart(7, '0');
}

function buildTrialId(state: SimulationState, outcome: TrialOutcome): string {
  const source =
    state.runSummary?.trialId ??
    [
      state.seed ?? 'demo',
      state.policy,
      state.stepCount,
      outcome,
      state.metrics.cumulativeReward.toFixed(4),
      state.metrics.burnedArea,
      state.metrics.infrastructureLost,
      state.metrics.resourcesConsumed,
    ].join('|');
  return 'trial-' + privacySafeHash(source);
}

export function buildRunReport(state: SimulationState): GuidedRunReport {
  const outcome = outcomeFromState(state);
  const totalCells = state.gridSize ** 2;
  const server = state.runSummary;
  const infrastructurePreserved =
    server?.finalMetrics?.infrastructurePreserved ??
    state.metrics.infrastructurePreserved;
  const infrastructureTotal =
    server?.finalMetrics?.infrastructureTotal ??
    state.metrics.infrastructureTotal;
  const burnedArea = server?.finalMetrics?.burnedArea ?? state.metrics.burnedArea;
  const resourcesConsumed =
    server?.finalMetrics?.resourcesConsumed ??
    state.metrics.resourcesConsumed;
  const totalReward =
    server?.finalMetrics?.cumulativeReward ??
    state.metrics.cumulativeReward;
  const terminationReason =
    server?.terminationReason ??
    state.terminationReason ??
    (state.metrics.burningCells === 0 ? 'fire_extinguished' : 'max_steps');
  const missionCompleted =
    server?.missionCompleted ?? terminationReason === 'fire_extinguished';
  const successful =
    server?.success ??
    (missionCompleted && infrastructurePreserved === infrastructureTotal);
  const lesson = recurringLesson(state, outcome);
  const defaultWhatHappened = [
    'The fire affected ' + formatCount(burnedArea, 'map square') + '.',
    infrastructurePreserved +
      ' of ' +
      infrastructureTotal +
      ' synthetic sites stayed safe.',
    'The agent used ' + formatCount(resourcesConsumed, 'response resource') + '.',
  ];
  const defaultRewardLesson =
    state.lastReward !== null && state.lastReward < 0
      ? 'The last move lost points. The total reward adds every step together.'
      : 'A better step loses fewer points. The total adds every step penalty together.';

  return {
    id: buildTrialId(state, outcome),
    outcome,
    missionCompleted,
    successful,
    title: server?.headline ?? outcomeTitle(outcome),
    whatHappened:
      server && server.whatHappened.length > 0
        ? server.whatHappened
        : defaultWhatHappened,
    rewardLesson: server?.reinforcement ?? defaultRewardLesson,
    improvements:
      server && server.improvements.length > 0
        ? server.improvements
        : fallbackImprovements(state, outcome),
    xpEarned: Math.max(0, Math.round(server?.xpEarned ?? xpForRun(state, outcome))),
    terminationReason,
    totalReward,
    stepReward: state.lastReward,
    infrastructurePreserved,
    infrastructureTotal,
    burnedArea,
    totalCells,
    resourcesConsumed,
    lessonCode: lesson.code,
    lesson: lesson.text,
    source: server ? 'server summary' : 'result-based summary',
  };
}

export function outcomeLabel(outcome: TrialOutcome): string {
  if (outcome === 'success') return 'Success';
  if (outcome === 'partial') return 'Partial';
  return 'Needs work';
}
