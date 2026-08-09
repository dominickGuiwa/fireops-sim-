import type { TrialOutcome, TrialStats } from '../types/simulation';
import type { GuidedRunReport } from './guidance';

const TRIAL_LOG_KEY = 'fireops-sim.practice-trials.v1';
const TUTORIAL_KEY = 'fireops-sim.tutorial-seen.v1';
const MAX_SAVED_TRIALS = 100;
const MAX_KNOWN_TRIAL_IDS = 500;

export interface PracticeTrial {
  id: string;
  completedAt: string;
  outcome: TrialOutcome;
  reward: number;
  infrastructurePreserved: number;
  infrastructureTotal: number;
  burnedArea: number;
  totalCells: number;
  resourcesConsumed: number;
  xpEarned: number;
  lessonCode: string;
  lesson: string;
}

export interface PracticeNotebookStats {
  allTimeTrials: number;
  successRate: number;
  averageReward: number | null;
  bestReward: number | null;
  totalXp: number;
  level: number;
  recurringLesson: string | null;
}

export interface PracticeAggregate {
  totalTrials: number;
  successes: number;
  totalReward: number;
  bestReward: number | null;
  totalXp: number;
  lessons: Record<string, { count: number; text: string }>;
}

export interface PracticeLog {
  version: 1;
  recent: PracticeTrial[];
  knownIds: string[];
  aggregate: PracticeAggregate;
}

const EMPTY_AGGREGATE: PracticeAggregate = {
  totalTrials: 0,
  successes: 0,
  totalReward: 0,
  bestReward: null,
  totalXp: 0,
  lessons: {},
};

function isTrialOutcome(value: unknown): value is TrialOutcome {
  return value === 'success' || value === 'partial' || value === 'needs_improvement';
}

function isPracticeTrial(value: unknown): value is PracticeTrial {
  if (typeof value !== 'object' || value === null) return false;
  const trial = value as Partial<PracticeTrial>;
  return (
    typeof trial.id === 'string' &&
    typeof trial.completedAt === 'string' &&
    isTrialOutcome(trial.outcome) &&
    typeof trial.reward === 'number' &&
    typeof trial.infrastructurePreserved === 'number' &&
    typeof trial.infrastructureTotal === 'number' &&
    typeof trial.burnedArea === 'number' &&
    typeof trial.totalCells === 'number' &&
    typeof trial.resourcesConsumed === 'number' &&
    typeof trial.xpEarned === 'number' &&
    typeof trial.lessonCode === 'string' &&
    typeof trial.lesson === 'string'
  );
}

function aggregateTrials(trials: PracticeTrial[]): PracticeAggregate {
  return trials.reduce<PracticeAggregate>(
    (aggregate, trial) => {
      const lesson = aggregate.lessons[trial.lessonCode] ?? {
        count: 0,
        text: trial.lesson,
      };
      return {
        totalTrials: aggregate.totalTrials + 1,
        successes: aggregate.successes + (trial.outcome === 'success' ? 1 : 0),
        totalReward: aggregate.totalReward + trial.reward,
        bestReward:
          aggregate.bestReward === null
            ? trial.reward
            : Math.max(aggregate.bestReward, trial.reward),
        totalXp: aggregate.totalXp + trial.xpEarned,
        lessons: {
          ...aggregate.lessons,
          [trial.lessonCode]: { count: lesson.count + 1, text: trial.lesson },
        },
      };
    },
    { ...EMPTY_AGGREGATE, lessons: {} },
  );
}

function isAggregate(value: unknown): value is PracticeAggregate {
  if (typeof value !== 'object' || value === null) return false;
  const aggregate = value as Partial<PracticeAggregate>;
  return (
    typeof aggregate.totalTrials === 'number' &&
    typeof aggregate.successes === 'number' &&
    typeof aggregate.totalReward === 'number' &&
    (aggregate.bestReward === null || typeof aggregate.bestReward === 'number') &&
    typeof aggregate.totalXp === 'number' &&
    typeof aggregate.lessons === 'object' &&
    aggregate.lessons !== null
  );
}

export function loadPracticeLog(): PracticeLog {
  try {
    const raw = window.localStorage.getItem(TRIAL_LOG_KEY);
    if (!raw) {
      return {
        version: 1,
        recent: [],
        knownIds: [],
        aggregate: { ...EMPTY_AGGREGATE, lessons: {} },
      };
    }
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      const recent = parsed.filter(isPracticeTrial).slice(0, MAX_SAVED_TRIALS);
      return {
        version: 1,
        recent,
        knownIds: recent.map((trial) => trial.id),
        aggregate: aggregateTrials(recent),
      };
    }
    if (typeof parsed === 'object' && parsed !== null) {
      const candidate = parsed as Partial<PracticeLog>;
      const recent = Array.isArray(candidate.recent)
        ? candidate.recent.filter(isPracticeTrial).slice(0, MAX_SAVED_TRIALS)
        : [];
      if (isAggregate(candidate.aggregate)) {
        return {
          version: 1,
          recent,
          knownIds: Array.isArray(candidate.knownIds)
            ? candidate.knownIds
                .filter((id): id is string => typeof id === 'string')
                .slice(0, MAX_KNOWN_TRIAL_IDS)
            : recent.map((trial) => trial.id),
          aggregate: candidate.aggregate,
        };
      }
    }
  } catch {
    // Fall through to an empty, in-memory notebook.
  }
  return {
    version: 1,
    recent: [],
    knownIds: [],
    aggregate: { ...EMPTY_AGGREGATE, lessons: {} },
  };
}

function writePracticeLog(log: PracticeLog): boolean {
  try {
    window.localStorage.setItem(TRIAL_LOG_KEY, JSON.stringify(log));
    return true;
  } catch {
    return false;
  }
}

export function savePracticeTrial(
  report: GuidedRunReport,
  current: PracticeLog,
): { log: PracticeLog; added: boolean } {
  if (current.knownIds.includes(report.id)) {
    return { log: current, added: false };
  }
  const trial: PracticeTrial = {
    id: report.id,
    completedAt: new Date().toISOString(),
    outcome: report.outcome,
    reward: report.totalReward,
    infrastructurePreserved: report.infrastructurePreserved,
    infrastructureTotal: report.infrastructureTotal,
    burnedArea: report.burnedArea,
    totalCells: report.totalCells,
    resourcesConsumed: report.resourcesConsumed,
    xpEarned: report.xpEarned,
    lessonCode: report.lessonCode,
    lesson: report.lesson,
  };
  const previousLesson = current.aggregate.lessons[trial.lessonCode] ?? {
    count: 0,
    text: trial.lesson,
  };
  const log: PracticeLog = {
    version: 1,
    recent: [trial, ...current.recent].slice(0, MAX_SAVED_TRIALS),
    knownIds: [trial.id, ...current.knownIds].slice(0, MAX_KNOWN_TRIAL_IDS),
    aggregate: {
      totalTrials: current.aggregate.totalTrials + 1,
      successes:
        current.aggregate.successes + (trial.outcome === 'success' ? 1 : 0),
      totalReward: current.aggregate.totalReward + trial.reward,
      bestReward:
        current.aggregate.bestReward === null
          ? trial.reward
          : Math.max(current.aggregate.bestReward, trial.reward),
      totalXp: current.aggregate.totalXp + trial.xpEarned,
      lessons: {
        ...current.aggregate.lessons,
        [trial.lessonCode]: {
          count: previousLesson.count + 1,
          text: trial.lesson,
        },
      },
    },
  };
  writePracticeLog(log);
  return { log, added: true };
}

export function clearPracticeLog(): PracticeLog {
  try {
    window.localStorage.removeItem(TRIAL_LOG_KEY);
    window.localStorage.removeItem(TUTORIAL_KEY);
  } catch {
    // Memory state is still cleared when browser storage is unavailable.
  }
  return {
    version: 1,
    recent: [],
    knownIds: [],
    aggregate: { ...EMPTY_AGGREGATE, lessons: {} },
  };
}

export function computePracticeStats(log: PracticeLog): PracticeNotebookStats {
  const aggregate = log.aggregate;
  if (aggregate.totalTrials === 0) {
    return {
      allTimeTrials: 0,
      successRate: 0,
      averageReward: null,
      bestReward: null,
      totalXp: 0,
      level: 1,
      recurringLesson: null,
    };
  }
  const recurringLesson =
    Object.values(aggregate.lessons).sort((left, right) => right.count - left.count)[0]
      ?.text ?? null;
  return {
    allTimeTrials: aggregate.totalTrials,
    successRate: aggregate.successes / aggregate.totalTrials,
    averageReward: aggregate.totalReward / aggregate.totalTrials,
    bestReward: aggregate.bestReward,
    totalXp: aggregate.totalXp,
    level: Math.floor(aggregate.totalXp / 250) + 1,
    recurringLesson,
  };
}

export function serverSessionLabel(stats: TrialStats | null): string | null {
  if (!stats || stats.trialsCompleted === null) return null;
  const successCount = stats.successes ?? 0;
  return (
    'This API session: ' +
    stats.trialsCompleted +
    ' completed, ' +
    successCount +
    ' successful.'
  );
}

export function tutorialWasSeen(): boolean {
  try {
    return window.localStorage.getItem(TUTORIAL_KEY) === 'seen';
  } catch {
    return false;
  }
}

export function markTutorialSeen(): void {
  try {
    window.localStorage.setItem(TUTORIAL_KEY, 'seen');
  } catch {
    // The tutorial can still close when browser storage is unavailable.
  }
}
