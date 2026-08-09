export const GRID_SIZE = 20;

export type TerrainType = 'vegetation' | 'road' | 'open' | 'water';
export type FireState = 'none' | 'burning' | 'burned';
export type InfrastructureType = 'medical' | 'power' | 'communications';
export type SensorStatus = 'online' | 'degraded' | 'offline';
export type EventSeverity = 'info' | 'warning' | 'critical' | 'success';
export type ConnectionStatus = 'connecting' | 'online' | 'offline';
export type TrialOutcome = 'success' | 'partial' | 'needs_improvement';
export type TrainingDifficulty = 'standard' | 'hard' | 'brutal';

export interface KnowledgeResult {
  rank: number;
  title: string;
  text: string;
  score: number;
}

export interface KnowledgeAnswer {
  query: string;
  results: KnowledgeResult[];
  answer: string | null;
  note: string | null;
}

export interface GridCell {
  row: number;
  col: number;
  terrain: TerrainType;
  fire: FireState;
  trueFire: FireState | null;
  infrastructure: InfrastructureType | null;
  observed: boolean;
  protection: number;
  intensity: number;
}

export interface WindState {
  direction: string;
  observedDirection: string | null;
  strength: number;
}

export interface SensorState {
  reliability: number;
  configuredReliability: number;
  status: SensorStatus;
  observedFraction: number;
  communicationDegraded: boolean;
  reconActive: boolean;
}

export interface ResourceState {
  available: number;
  initial: number;
  consumed: number;
  unavailable: number;
}

export interface SimulationMetrics {
  cumulativeReward: number;
  burnedArea: number;
  burningCells: number;
  infrastructurePreserved: number;
  infrastructureTotal: number;
  infrastructureLost: number;
  resourcesConsumed: number;
}

export interface SimulationEvent {
  id: string;
  step: number;
  type: string;
  message: string;
  severity: EventSeverity;
}

export interface AgentDecision {
  seenSummary: string | null;
  visibleCells: number | null;
  visibleFireCells: number | null;
  hiddenCells: number | null;
  action: string | null;
  actionLabel: string | null;
  decisionSource: 'policy' | 'manual' | null;
  reason: string | null;
  resourcesBeforeAction: number | null;
  rewardDelta: number | null;
  rewardBreakdown: RewardBreakdown | null;
  confidence: number | null;
  explored: boolean | null;
  explorationRate: number | null;
  source: 'backend' | 'local_demo';
}

export type AgentFailureType =
  | 'perception_gap'
  | 'misdirected_protection'
  | 'idle_while_burning'
  | 'resource_exhausted_request'
  | 'wind_misread';

export interface AgentFailure {
  id: string;
  step: number;
  policy: string;
  failureType: AgentFailureType;
  message: string;
  hiddenFireCells: number;
  visibleFireCells: number;
  trueBurningCells: number;
  source: 'backend' | 'local_demo';
}

export interface LearningEpisode {
  episode: number;
  reward: number;
  outcome: TrialOutcome;
  epsilon: number;
  source: 'interactive' | 'training';
}

export interface LearningStats {
  episodesTrained: number;
  epsilon: number;
  learningRate: number;
  discount: number;
  updatesApplied: number;
  statesVisited: number;
  rewardHistory: LearningEpisode[];
  earlyAverage: number | null;
  recentAverage: number | null;
  improvement: number | null;
  note: string | null;
  source: 'backend';
}

export interface RewardBreakdown {
  newSpreadPenalty: number;
  infrastructureLossPenalty: number;
  resourceUsePenalty: number;
  activeFirePenalty: number;
  total: number;
}

export interface TrialStats {
  scope: 'server_session';
  trialsStarted: number | null;
  trialsCompleted: number | null;
  successes: number | null;
  partials: number | null;
  needsImprovement: number | null;
  totalXp: number | null;
  currentLevel: number | null;
  averageCompletedReward: number | null;
  source: 'backend';
}

export interface RunActionCounts {
  hold: number;
  protectNorth: number;
  protectSouth: number;
  protectEast: number;
  protectWest: number;
  recon: number;
}

export interface RunFinalMetrics {
  cumulativeReward: number | null;
  burnedArea: number | null;
  burningCells: number | null;
  infrastructurePreserved: number | null;
  infrastructureTotal: number | null;
  infrastructureLost: number | null;
  resourcesConsumed: number | null;
}

export interface RunSummary {
  trialId: string | null;
  missionCompleted: boolean | null;
  success: boolean | null;
  outcome: TrialOutcome | null;
  headline: string | null;
  whatHappened: string[];
  reinforcement: string | null;
  improvements: string[];
  xpEarned: number | null;
  actionCounts: RunActionCounts | null;
  terminationReason: string | null;
  finalMetrics: RunFinalMetrics | null;
  source: 'backend';
}

export interface SimulationState {
  seed: number | null;
  gridSize: number;
  maxSteps: number | null;
  stepCount: number;
  cells: GridCell[];
  wind: WindState;
  sensor: SensorState;
  resources: ResourceState;
  metrics: SimulationMetrics;
  lastReward: number | null;
  policy: string;
  currentAction: string;
  agentDecision: AgentDecision | null;
  trialStats: TrialStats | null;
  runSummary: RunSummary | null;
  agentFailures: AgentFailure[];
  learning: LearningStats | null;
  events: SimulationEvent[];
  done: boolean;
  terminationReason: string | null;
}

export interface PolicyDefinition {
  id: string;
  label: string;
  description: string;
  category: 'baseline' | 'heuristic' | 'learned';
  nativeAvailable: boolean;
}

export interface ActionDefinition {
  id: number;
  value: string;
  label: string;
}

export interface ResetOptions {
  seed?: number;
  policy?: string;
  sensorReliability?: number;
}

export const ACTIONS: ActionDefinition[] = [
  { id: 0, value: 'HOLD', label: 'Hold' },
  { id: 1, value: 'PROTECT_NORTH', label: 'Protect north' },
  { id: 2, value: 'PROTECT_SOUTH', label: 'Protect south' },
  { id: 3, value: 'PROTECT_EAST', label: 'Protect east' },
  { id: 4, value: 'PROTECT_WEST', label: 'Protect west' },
  { id: 5, value: 'RECON', label: 'Recon' },
];

export const DEFAULT_POLICIES: PolicyDefinition[] = [
  {
    id: 'rule',
    label: 'Rule Agent',
    description: 'Uses visible fire and simple rules to protect threatened sites.',
    category: 'heuristic',
    nativeAvailable: true,
  },
  {
    id: 'random',
    label: 'Random',
    description: 'Picks a move at random so you can compare it with a fixed strategy.',
    category: 'baseline',
    nativeAvailable: true,
  },
  {
    id: 'learning',
    label: 'Learning (Q-table)',
    description:
      'A transparent value table that really learns from rewards on the server. Offline it plays like the Rule Agent.',
    category: 'learned',
    nativeAvailable: true,
  },
  {
    id: 'ppo_nominal',
    label: 'PPO Nominal',
    description: 'A saved policy made for mostly reliable maps.',
    category: 'learned',
    nativeAvailable: false,
  },
  {
    id: 'ppo_robust',
    label: 'PPO Robust',
    description: 'A saved policy made for maps with synthetic sensor problems.',
    category: 'learned',
    nativeAvailable: false,
  },
];
