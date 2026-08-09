import {
  ACTIONS,
  DEFAULT_POLICIES,
  GRID_SIZE,
  type EventSeverity,
  type FireState,
  type GridCell,
  type InfrastructureType,
  type AgentDecision,
  type AgentFailure,
  type AgentFailureType,
  type LearningEpisode,
  type LearningStats,
  type PolicyDefinition,
  type RewardBreakdown,
  type RunActionCounts,
  type RunFinalMetrics,
  type RunSummary,
  type SimulationEvent,
  type SimulationMetrics,
  type SimulationState,
  type TerrainType,
  type TrialOutcome,
  type TrialStats,
} from '../types/simulation';

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function firstRecord(...values: unknown[]): UnknownRecord {
  return values.find(isRecord) ?? {};
}

function firstArray(...values: unknown[]): unknown[] {
  return values.find(Array.isArray) ?? [];
}

function firstNumber(fallback: number, ...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return fallback;
}

function firstString(fallback: string, ...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return fallback;
}

function optionalNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}

function optionalString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return null;
}

function optionalBoolean(...values: unknown[]): boolean | null {
  for (const value of values) {
    if (typeof value === 'boolean') return value;
  }
  return null;
}

function stringList(...values: unknown[]): string[] {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value
        .filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
        .map((item) => item.trim());
    }
    if (typeof value === 'string' && value.trim()) return [value.trim()];
  }
  return [];
}

export function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeReliability(value: unknown, fallback: number): number {
  const parsed = firstNumber(fallback, value);
  return clamp(parsed > 1 ? parsed / 100 : parsed);
}

function normalizeObservedDirection(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  const normalized = value.trim().toUpperCase();
  return normalized === 'UNKNOWN' || normalized === 'NONE' || normalized === 'UNAVAILABLE'
    ? null
    : normalized;
}

function normalizeTerrain(value: unknown): TerrainType {
  const text = typeof value === 'string' ? value.toLowerCase() : '';
  if (text.includes('road')) return 'road';
  if (text.includes('open') || text.includes('ground') || text.includes('clear')) return 'open';
  if (text.includes('water')) return 'water';
  return 'vegetation';
}

function normalizeFire(value: unknown, descriptor: string): FireState {
  const text = typeof value === 'string' ? value.toLowerCase() : descriptor;
  if (
    text === 'unburned' ||
    text === 'unburnt' ||
    text === 'none' ||
    text === 'unknown' ||
    text === 'inactive'
  ) {
    return 'none';
  }
  if (text.includes('burned') || text.includes('burnt') || text.includes('ash')) return 'burned';
  if (text.includes('burning') || text.includes('active') || text === 'fire') return 'burning';
  if (value === true || value === 1) return 'burning';
  if (value === 2) return 'burned';
  return 'none';
}

function normalizeInfrastructure(value: unknown, descriptor: string): InfrastructureType | null {
  const text = typeof value === 'string' ? value.toLowerCase() : descriptor;
  if (text.includes('med') || text.includes('hospital')) return 'medical';
  if (text.includes('power') || text.includes('electric') || text.includes('grid')) return 'power';
  if (text.includes('comm') || text.includes('radio') || text.includes('tower')) {
    return 'communications';
  }
  return null;
}

function defaultCell(row: number, col: number): GridCell {
  return {
    row,
    col,
    terrain: 'vegetation',
    fire: 'none',
    trueFire: null,
    infrastructure: null,
    observed: true,
    protection: 0,
    intensity: 0,
  };
}

function normalizeCell(raw: unknown, index: number, gridSize: number): GridCell {
  const fallbackRow = Math.floor(index / gridSize);
  const fallbackCol = index % gridSize;

  if (!isRecord(raw)) {
    const descriptor = typeof raw === 'string' ? raw.toLowerCase() : '';
    const numericType = typeof raw === 'number' ? raw : -1;
    const fire =
      numericType === 1
        ? 'burning'
        : numericType === 2
          ? 'burned'
          : normalizeFire(raw, descriptor);
    return {
      ...defaultCell(fallbackRow, fallbackCol),
      terrain:
        numericType === 3 || descriptor.includes('road')
          ? 'road'
          : normalizeTerrain(descriptor),
      fire,
      trueFire: null,
      infrastructure:
        numericType === 4
          ? 'medical'
          : numericType === 5
            ? 'power'
            : numericType === 6
              ? 'communications'
              : normalizeInfrastructure(raw, descriptor),
    };
  }

  const descriptor = firstString(
    '',
    raw.type,
    raw.state,
    raw.cell_type,
    raw.terrain,
  ).toLowerCase();
  const row = Math.round(firstNumber(fallbackRow, raw.row, raw.r, raw.y));
  const col = Math.round(firstNumber(fallbackCol, raw.col, raw.column, raw.c, raw.x));
  const protectionValue =
    typeof raw.protection === 'boolean'
      ? raw.protection
        ? 1
        : 0
      : firstNumber(0, raw.protection, raw.protected);

  const fire = normalizeFire(raw.fire ?? raw.fire_state, descriptor);
  return {
    row: clamp(row, 0, gridSize - 1),
    col: clamp(col, 0, gridSize - 1),
    terrain: normalizeTerrain(raw.terrain ?? raw.type),
    fire,
    trueFire:
      raw.true_fire === undefined && raw.trueFire === undefined
        ? null
        : normalizeFire(raw.true_fire ?? raw.trueFire, ''),
    infrastructure: normalizeInfrastructure(
      raw.infrastructure ?? raw.facility ?? raw.asset,
      descriptor,
    ),
    observed:
      typeof raw.observed === 'boolean'
        ? raw.observed
        : typeof raw.visible === 'boolean'
          ? raw.visible
          : true,
    protection: clamp(protectionValue),
    intensity: clamp(firstNumber(0.65, raw.intensity, raw.fire_intensity)),
  };
}

function normalizeCells(
  root: UnknownRecord,
  gridSize: number,
  fallbackCells: GridCell[],
): GridCell[] {
  let source = firstArray(root.cells);
  if (source.length === 0 && Array.isArray(root.grid)) {
    const grid = root.grid;
    source = grid.flatMap((row, rowIndex) =>
      Array.isArray(row)
        ? row.map((cell, colIndex) =>
            isRecord(cell) ? { ...cell, row: rowIndex, col: colIndex } : cell,
          )
        : [],
    );
  }

  if (source.length === 0 && fallbackCells.length === gridSize * gridSize) {
    return fallbackCells.map((cell) => ({ ...cell }));
  }
  const cells = Array.from({ length: gridSize * gridSize }, (_, index) =>
    defaultCell(Math.floor(index / gridSize), index % gridSize),
  );
  source.forEach((raw, index) => {
    const cell = normalizeCell(raw, index, gridSize);
    const targetIndex = cell.row * gridSize + cell.col;
    if (targetIndex >= 0 && targetIndex < cells.length) {
      cells[targetIndex] = cell;
    }
  });
  return cells;
}

function normalizeSeverity(value: unknown): EventSeverity {
  const text = typeof value === 'string' ? value.toLowerCase() : '';
  if (text.includes('critical') || text.includes('error') || text.includes('danger')) {
    return 'critical';
  }
  if (text.includes('warn') || text.includes('degrad')) return 'warning';
  if (text.includes('success') || text.includes('recover') || text.includes('online')) {
    return 'success';
  }
  return 'info';
}

function normalizeEvents(root: UnknownRecord, fallback: SimulationEvent[]): SimulationEvent[] {
  const rawEvents = firstArray(root.events, root.event_log, root.log);
  if (rawEvents.length === 0) return fallback;
  return rawEvents
    .map((raw, index): SimulationEvent | null => {
      if (typeof raw === 'string') {
        return {
          id: 'event-' + index + '-' + raw,
          step: firstNumber(0, root.step_count),
          type: 'system',
          message: raw,
          severity: 'info',
        };
      }
      if (!isRecord(raw)) return null;
      const message = firstString('', raw.message, raw.detail, raw.event);
      if (!message) return null;
      const step = Math.round(firstNumber(0, raw.step, raw.tick));
      const type = firstString('system', raw.type, raw.category);
      return {
        id: firstString('event-' + step + '-' + index + '-' + message, raw.id),
        step,
        type,
        message,
        severity: normalizeSeverity(raw.severity ?? raw.level ?? type),
      };
    })
    .filter((event): event is SimulationEvent => event !== null)
    .slice(-60);
}

function normalizeAgentDecision(value: unknown): AgentDecision | null {
  if (!isRecord(value)) return null;
  const rawAction =
    value.chosen_action ?? value.action ?? value.picked_action ?? value.selected_action;
  const action =
    typeof rawAction === 'number' || typeof rawAction === 'string'
      ? normalizeAction(rawAction)
      : null;
  const confidenceValue = optionalNumber(value.confidence, value.action_confidence);
  const rawDecisionSource = optionalString(value.decision_source);
  const decisionSource =
    rawDecisionSource === 'policy' || rawDecisionSource === 'manual'
      ? rawDecisionSource
      : null;
  const rewardBreakdown = normalizeRewardBreakdown(value.reward_breakdown);
  const decision: AgentDecision = {
    seenSummary: optionalString(
      value.seen_summary,
      value.observation_summary,
      value.what_agent_saw,
    ),
    visibleCells: optionalNumber(
      value.visible_cells,
      value.visible_cell_count,
      value.observed_cells,
    ),
    visibleFireCells: optionalNumber(
      value.visible_fire_cells,
      value.visible_fire_count,
      value.visible_burning_cells,
      value.observed_fire_cells,
    ),
    hiddenCells: optionalNumber(value.hidden_cells, value.hidden_cell_count),
    action,
    actionLabel: optionalString(value.action_label),
    decisionSource,
    reason: optionalString(value.reason, value.rationale, value.explanation),
    resourcesBeforeAction: optionalNumber(value.resources_before_action),
    rewardDelta: optionalNumber(value.reward_delta, value.reward_change, value.step_reward),
    rewardBreakdown,
    confidence:
      confidenceValue === null
        ? null
        : clamp(confidenceValue > 1 ? confidenceValue / 100 : confidenceValue),
    explored: optionalBoolean(value.explored),
    explorationRate: optionalNumber(value.exploration_rate, value.epsilon),
    source: 'backend',
  };
  return Object.entries(decision).some(
    ([key, item]) => key !== 'source' && item !== null,
  )
    ? decision
    : null;
}

const FAILURE_TYPES: AgentFailureType[] = [
  'perception_gap',
  'misdirected_protection',
  'idle_while_burning',
  'resource_exhausted_request',
  'wind_misread',
];

function normalizeAgentFailures(value: unknown): AgentFailure[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((raw, index): AgentFailure | null => {
      if (!isRecord(raw)) return null;
      const failureType = optionalString(raw.failure_type, raw.type);
      const message = optionalString(raw.message);
      if (!message || !FAILURE_TYPES.includes(failureType as AgentFailureType)) {
        return null;
      }
      const step = Math.max(0, Math.round(firstNumber(0, raw.step)));
      return {
        id: 'failure-' + step + '-' + index + '-' + failureType,
        step,
        policy: firstString('policy', raw.policy),
        failureType: failureType as AgentFailureType,
        message,
        hiddenFireCells: Math.max(0, firstNumber(0, raw.hidden_fire_cells)),
        visibleFireCells: Math.max(0, firstNumber(0, raw.visible_fire_cells)),
        trueBurningCells: Math.max(0, firstNumber(0, raw.true_burning_cells)),
        source: 'backend',
      };
    })
    .filter((failure): failure is AgentFailure => failure !== null)
    .slice(-25);
}

function normalizeLearningStats(value: unknown): LearningStats | null {
  if (!isRecord(value)) return null;
  const episodesTrained = optionalNumber(value.episodes_trained);
  const epsilon = optionalNumber(value.epsilon);
  if (episodesTrained === null || epsilon === null) return null;
  const rewardHistory: LearningEpisode[] = Array.isArray(value.reward_history)
    ? value.reward_history
        .map((raw): LearningEpisode | null => {
          if (!isRecord(raw)) return null;
          const episode = optionalNumber(raw.episode);
          const reward = optionalNumber(raw.reward);
          if (episode === null || reward === null) return null;
          const outcome = normalizeTrialOutcome(raw.outcome) ?? 'needs_improvement';
          return {
            episode: Math.round(episode),
            reward,
            outcome,
            epsilon: clamp(firstNumber(0, raw.epsilon)),
            source: raw.source === 'interactive' ? 'interactive' : 'training',
          };
        })
        .filter((item): item is LearningEpisode => item !== null)
    : [];
  return {
    episodesTrained: Math.max(0, Math.round(episodesTrained)),
    epsilon: clamp(epsilon),
    learningRate: clamp(firstNumber(0, value.learning_rate)),
    discount: clamp(firstNumber(0, value.discount)),
    updatesApplied: Math.max(0, Math.round(firstNumber(0, value.updates_applied))),
    statesVisited: Math.max(0, Math.round(firstNumber(0, value.states_visited))),
    rewardHistory,
    earlyAverage: optionalNumber(value.early_average),
    recentAverage: optionalNumber(value.recent_average),
    improvement: optionalNumber(value.improvement),
    note: optionalString(value.note),
    source: 'backend',
  };
}

function normalizeRewardBreakdown(value: unknown): RewardBreakdown | null {
  if (!isRecord(value)) return null;
  const newSpreadPenalty = optionalNumber(value.new_spread_penalty);
  const infrastructureLossPenalty = optionalNumber(value.infrastructure_loss_penalty);
  const resourceUsePenalty = optionalNumber(value.resource_use_penalty);
  const activeFirePenalty = optionalNumber(value.active_fire_penalty);
  const total = optionalNumber(value.total);
  if (
    newSpreadPenalty === null ||
    infrastructureLossPenalty === null ||
    resourceUsePenalty === null ||
    activeFirePenalty === null ||
    total === null
  ) {
    return null;
  }
  return {
    newSpreadPenalty,
    infrastructureLossPenalty,
    resourceUsePenalty,
    activeFirePenalty,
    total,
  };
}

function normalizeTrialOutcome(value: unknown): TrialOutcome | null {
  if (typeof value !== 'string') return null;
  const normalized = value.toLowerCase().replaceAll('-', '_').replaceAll(' ', '_');
  if (normalized === 'success' || normalized === 'successful' || normalized === 'win') {
    return 'success';
  }
  if (normalized === 'partial' || normalized === 'partial_success') return 'partial';
  if (
    normalized === 'needs_work' ||
    normalized === 'needs_improvement' ||
    normalized === 'failure' ||
    normalized === 'failed' ||
    normalized === 'loss'
  ) {
    return 'needs_improvement';
  }
  return null;
}

function normalizeTrialStats(value: unknown): TrialStats | null {
  if (!isRecord(value)) return null;
  const stats: TrialStats = {
    scope: 'server_session',
    trialsStarted: optionalNumber(
      value.trials_started,
      value.all_time_trials,
      value.total_trials,
    ),
    trialsCompleted: optionalNumber(value.trials_completed, value.completed_trials),
    successes: optionalNumber(value.successes, value.success_count),
    partials: optionalNumber(value.partials, value.partial_count),
    needsImprovement: optionalNumber(
      value.needs_improvement,
      value.needs_improvement_count,
    ),
    totalXp: optionalNumber(value.total_experience_points, value.total_xp, value.xp),
    currentLevel: optionalNumber(value.current_level, value.level),
    averageCompletedReward: optionalNumber(
      value.average_completed_reward,
      value.average_reward,
      value.avg_reward,
    ),
    source: 'backend',
  };
  return Object.entries(stats).some(
    ([key, item]) => key !== 'source' && key !== 'scope' && item !== null,
  )
    ? stats
    : null;
}

function normalizeActionCounts(value: unknown): RunActionCounts | null {
  if (!isRecord(value)) return null;
  return {
    hold: Math.max(0, optionalNumber(value.hold) ?? 0),
    protectNorth: Math.max(0, optionalNumber(value.protect_north) ?? 0),
    protectSouth: Math.max(0, optionalNumber(value.protect_south) ?? 0),
    protectEast: Math.max(0, optionalNumber(value.protect_east) ?? 0),
    protectWest: Math.max(0, optionalNumber(value.protect_west) ?? 0),
    recon: Math.max(0, optionalNumber(value.recon) ?? 0),
  };
}

function normalizeFinalMetrics(value: unknown): RunFinalMetrics | null {
  if (!isRecord(value)) return null;
  const normalized: RunFinalMetrics = {
    cumulativeReward: optionalNumber(value.cumulative_reward, value.reward),
    burnedArea: optionalNumber(value.burned_area),
    burningCells: optionalNumber(value.burning_cells),
    infrastructurePreserved: optionalNumber(value.infrastructure_preserved),
    infrastructureTotal: optionalNumber(value.infrastructure_total),
    infrastructureLost: optionalNumber(value.infrastructure_lost),
    resourcesConsumed: optionalNumber(value.resources_consumed),
  };
  return Object.values(normalized).some((item) => item !== null) ? normalized : null;
}

function normalizeRunSummary(value: unknown): RunSummary | null {
  if (!isRecord(value)) return null;
  const summary: RunSummary = {
    trialId: optionalString(value.trial_id, value.id, value.run_id),
    missionCompleted: optionalBoolean(value.fire_extinguished, value.mission_completed),
    success: optionalBoolean(value.success),
    outcome: normalizeTrialOutcome(value.outcome ?? value.status ?? value.result),
    headline: optionalString(value.headline, value.title, value.summary),
    whatHappened: stringList(
      value.what_happened,
      value.highlights,
      value.events,
    ),
    reinforcement: optionalString(
      value.reinforcement,
      value.reward_lesson,
      value.what_worked,
      value.positive_feedback,
    ),
    improvements: stringList(
      value.improvements,
      value.next_steps,
      value.try_next,
    ),
    xpEarned: optionalNumber(value.xp_earned, value.xp, value.experience),
    actionCounts: normalizeActionCounts(value.action_counts),
    terminationReason: optionalString(value.termination_reason),
    finalMetrics: normalizeFinalMetrics(value.final_metrics),
    source: 'backend',
  };
  return Object.entries(summary).some(
    ([key, item]) =>
      key !== 'source' &&
      item !== null &&
      (!Array.isArray(item) || item.length > 0),
  )
    ? summary
    : null;
}

function deriveMetrics(cells: GridCell[], previous?: SimulationMetrics): SimulationMetrics {
  const burnedArea = cells.filter(
    (cell) => (cell.trueFire ?? cell.fire) !== 'none',
  ).length;
  const burningCells = cells.filter((cell) => (cell.trueFire ?? cell.fire) === 'burning').length;
  const infrastructureCells = cells.filter((cell) => cell.infrastructure !== null);
  const infrastructureLost = infrastructureCells.filter(
    (cell) => (cell.trueFire ?? cell.fire) !== 'none',
  ).length;
  return {
    cumulativeReward: previous?.cumulativeReward ?? 0,
    burnedArea,
    burningCells,
    infrastructurePreserved: infrastructureCells.length - infrastructureLost,
    infrastructureTotal: infrastructureCells.length,
    infrastructureLost,
    resourcesConsumed: previous?.resourcesConsumed ?? 0,
  };
}

export function normalizeSimulationState(
  payload: unknown,
  fallback: SimulationState,
): SimulationState {
  const envelope = isRecord(payload) ? payload : {};
  const root = firstRecord(envelope.state, envelope.simulation, envelope);
  const requestedSize = Math.round(firstNumber(fallback.gridSize, root.grid_size, root.size));
  const gridSize = requestedSize > 0 && requestedSize <= 50 ? requestedSize : GRID_SIZE;
  const cells = normalizeCells(root, gridSize, fallback.cells);
  const wind = firstRecord(root.wind, root.weather);
  const sensor = firstRecord(root.sensor, root.sensors, root.observation);
  const resources = firstRecord(root.resources, root.response_resources);
  const metrics = firstRecord(root.metrics, root.score);
  const derived = deriveMetrics(cells, fallback.metrics);
  const initialResources = Math.max(
    0,
    firstNumber(fallback.resources.initial, resources.initial, resources.total),
  );
  const availableResources = Math.max(
    0,
    firstNumber(fallback.resources.available, resources.available, resources.remaining),
  );
  const infrastructureTotal = Math.max(
    0,
    firstNumber(
      derived.infrastructureTotal,
      metrics.infrastructure_total,
      root.infrastructure_total,
    ),
  );
  const infrastructureLost = Math.max(
    0,
    firstNumber(
      derived.infrastructureLost,
      metrics.infrastructure_lost,
      root.infrastructure_lost,
    ),
  );
  const reliability = normalizeReliability(
    sensor.reliability ?? root.sensor_reliability,
    fallback.sensor.reliability,
  );
  const observedFraction = normalizeReliability(
    sensor.observed_fraction ?? root.observed_fraction,
    cells.filter((cell) => cell.observed).length / Math.max(1, cells.length),
  );
  const sensorStatusText = firstString(
    reliability < 0.4 ? 'offline' : reliability < 0.78 ? 'degraded' : 'online',
    sensor.status,
  ).toLowerCase();
  const trueWindDirection = firstString(
    fallback.wind.direction,
    wind.direction,
    root.wind_direction,
  ).toUpperCase();
  let observedWindDirection = fallback.wind.observedDirection;
  if ('observed_direction' in wind) {
    observedWindDirection = normalizeObservedDirection(wind.observed_direction);
  } else if ('observedDirection' in wind) {
    observedWindDirection = normalizeObservedDirection(wind.observedDirection);
  } else if ('observed_wind_direction' in root) {
    observedWindDirection = normalizeObservedDirection(root.observed_wind_direction);
  } else {
    observedWindDirection = trueWindDirection;
  }
  const seed = optionalNumber(root.seed, root.simulation_seed);

  return {
    seed: seed === null ? fallback.seed : Math.round(seed),
    gridSize,
    maxSteps:
      optionalNumber(root.max_steps) === null
        ? fallback.maxSteps
        : Math.max(1, Math.round(optionalNumber(root.max_steps) ?? 1)),
    stepCount: Math.max(
      0,
      Math.round(firstNumber(fallback.stepCount, root.step_count, root.step, root.tick)),
    ),
    cells,
    wind: {
      direction: trueWindDirection,
      observedDirection: observedWindDirection,
      strength: Math.max(
        0,
        firstNumber(fallback.wind.strength, wind.strength, wind.speed, root.wind_strength),
      ),
    },
    sensor: {
      reliability,
      configuredReliability: normalizeReliability(
        sensor.configured_reliability,
        reliability,
      ),
      status: sensorStatusText.includes('off')
        ? 'offline'
        : sensorStatusText.includes('degrad')
          ? 'degraded'
          : 'online',
      observedFraction,
      communicationDegraded: Boolean(
        sensor.communication_degraded ??
          sensor.comms_degraded ??
          root.communication_degraded ??
          false,
      ),
      reconActive: Boolean(sensor.recon_active ?? false),
    },
    resources: {
      available: availableResources,
      initial: initialResources,
      consumed: Math.max(
        0,
        firstNumber(
          initialResources - availableResources,
          resources.consumed,
          metrics.resources_consumed,
        ),
      ),
      unavailable: Math.max(0, firstNumber(0, resources.unavailable)),
    },
    metrics: {
      cumulativeReward: firstNumber(
        fallback.metrics.cumulativeReward,
        metrics.cumulative_reward,
        metrics.reward,
        root.cumulative_reward,
      ),
      burnedArea: Math.max(
        0,
        firstNumber(derived.burnedArea, metrics.burned_area, root.burned_area),
      ),
      burningCells: Math.max(
        0,
        firstNumber(derived.burningCells, metrics.burning_cells, root.burning_cells),
      ),
      infrastructurePreserved: Math.max(
        0,
        firstNumber(
          infrastructureTotal - infrastructureLost,
          metrics.infrastructure_preserved,
          root.infrastructure_preserved,
        ),
      ),
      infrastructureTotal,
      infrastructureLost,
      resourcesConsumed: Math.max(
        0,
        firstNumber(
          initialResources - availableResources,
          metrics.resources_consumed,
          resources.consumed,
        ),
      ),
    },
    lastReward: optionalNumber(root.last_reward, root.step_reward),
    policy: firstString(fallback.policy, root.policy, root.current_policy),
    currentAction: normalizeAction(
      firstString(fallback.currentAction, root.current_action, root.action),
    ),
    agentDecision: normalizeAgentDecision(root.agent_decision),
    trialStats: normalizeTrialStats(root.trial_stats),
    runSummary: normalizeRunSummary(root.run_summary),
    agentFailures: normalizeAgentFailures(root.agent_failures),
    learning: normalizeLearningStats(root.learning) ?? fallback.learning,
    events: normalizeEvents(root, fallback.events),
    done: Boolean(root.done ?? root.terminated ?? false),
    terminationReason: optionalString(
      root.termination_reason,
      root.terminationReason,
    ),
  };
}

export function normalizeAction(action: string | number): string {
  if (typeof action === 'number') {
    return ACTIONS.find((item) => item.id === action)?.value ?? 'HOLD';
  }
  const normalized = action.trim().toUpperCase().replaceAll(' ', '_');
  return ACTIONS.some((item) => item.value === normalized) ? normalized : action || 'HOLD';
}

export function actionLabel(action: string): string {
  return ACTIONS.find((item) => item.value === normalizeAction(action))?.label ?? action;
}

export function normalizePolicies(payload: unknown): PolicyDefinition[] {
  const envelope = isRecord(payload) ? payload : {};
  const list = Array.isArray(payload) ? payload : firstArray(envelope.policies, envelope.items);
  if (list.length === 0) return DEFAULT_POLICIES;
  const policies = list
    .map((item): PolicyDefinition | null => {
      if (typeof item === 'string') {
        const known = DEFAULT_POLICIES.find((policy) => policy.id === item);
        return (
          known ?? {
            id: item,
            label: item.replaceAll('_', ' '),
            description: 'Simulation policy exposed by the connected API.',
            category: 'heuristic',
            nativeAvailable: true,
          }
        );
      }
      if (!isRecord(item)) return null;
      const id = firstString('', item.id, item.value, item.name);
      if (!id) return null;
      const known = DEFAULT_POLICIES.find((policy) => policy.id === id);
      const rawCategory = firstString(known?.category ?? 'heuristic', item.category);
      const category: PolicyDefinition['category'] =
        rawCategory === 'baseline' || rawCategory === 'learned' ? rawCategory : 'heuristic';
      return {
        id,
        label: firstString(known?.label ?? id.replaceAll('_', ' '), item.label, item.name),
        description: firstString(
          known?.description ?? 'Simulation policy exposed by the connected API.',
          item.description,
        ),
        category,
        nativeAvailable:
          typeof item.native_available === 'boolean'
            ? item.native_available
            : known?.nativeAvailable ?? true,
      };
    })
    .filter((policy): policy is PolicyDefinition => policy !== null);
  return policies.length > 0 ? policies : DEFAULT_POLICIES;
}

function hash01(row: number, col: number, seed: number): number {
  const value = Math.sin(row * 91.733 + col * 37.719 + seed * 17.113) * 43_758.5453;
  return value - Math.floor(value);
}

function demoMetrics(
  cells: GridCell[],
  reward = 0,
  resourcesConsumed = 0,
): SimulationMetrics {
  const base = deriveMetrics(cells);
  return { ...base, cumulativeReward: reward, resourcesConsumed };
}

export function createDemoState(
  seed = 11,
  policy = 'rule',
  reliability = 0.82,
): SimulationState {
  const infrastructureByPosition: Record<string, InfrastructureType> = {
    '3:4': 'medical',
    '15:16': 'power',
    '6:14': 'communications',
  };
  const burning = new Set(['8:7', '8:8', '9:7', '9:8', '10:8', '10:9', '11:9']);
  const burned = new Set(['8:5', '8:6', '9:5', '9:6', '10:6', '10:7', '11:7', '11:8']);
  const cells = Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, index): GridCell => {
    const row = Math.floor(index / GRID_SIZE);
    const col = index % GRID_SIZE;
    const key = row + ':' + col;
    const road = row === 4 || col === 14;
    const fire: FireState = burning.has(key) ? 'burning' : burned.has(key) ? 'burned' : 'none';
    return {
      row,
      col,
      terrain: road ? 'road' : hash01(row, col, seed) > 0.92 ? 'open' : 'vegetation',
      fire,
      trueFire: fire,
      infrastructure: infrastructureByPosition[key] ?? null,
      observed: hash01(row, col, seed + 5) <= reliability,
      protection: 0,
      intensity: fire === 'burning' ? 0.34 + hash01(row, col, seed + 2) * 0.5 : 0,
    };
  });

  return {
    seed,
    gridSize: GRID_SIZE,
    maxSteps: 120,
    stepCount: 0,
    cells,
    wind: { direction: 'ENE', observedDirection: 'ENE', strength: 0.64 },
    sensor: {
      reliability,
      configuredReliability: reliability,
      status: reliability < 0.45 ? 'offline' : reliability < 0.78 ? 'degraded' : 'online',
      observedFraction: cells.filter((cell) => cell.observed).length / cells.length,
      communicationDegraded: false,
      reconActive: false,
    },
    resources: { available: 8, initial: 8, consumed: 0, unavailable: 0 },
    metrics: demoMetrics(cells),
    lastReward: null,
    policy,
    currentAction: 'HOLD',
    agentDecision: null,
    trialStats: null,
    runSummary: null,
    agentFailures: [],
    learning: null,
    events: [
      {
        id: 'demo-ready',
        step: 0,
        type: 'scenario',
        message: 'Synthetic trial ready: keep all three sites safe and help the fire go out.',
        severity: 'info',
      },
      {
        id: 'demo-agent-view',
        step: 0,
        type: 'sensor',
        message: 'Agent View is ready. Hidden squares are not visible to the policy.',
        severity: 'info',
      },
    ],
    done: false,
    terminationReason: null,
  };
}

function neighborsOf(cell: GridCell, cells: GridCell[], gridSize: number): GridCell[] {
  const offsets = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  return offsets
    .map(([rowDelta, colDelta]) => {
      const row = cell.row + rowDelta;
      const col = cell.col + colDelta;
      return row >= 0 && col >= 0 && row < gridSize && col < gridSize
        ? cells[row * gridSize + col]
        : undefined;
    })
    .filter((neighbor): neighbor is GridCell => neighbor !== undefined);
}

function visibleFireSectorScores(state: SimulationState): Array<[string, number]> {
  const burning = state.cells.filter(
    (cell) => cell.observed && cell.fire === 'burning',
  );
  return [
    ['PROTECT_NORTH', burning.reduce((sum, cell) => sum + state.gridSize - cell.row, 0)],
    ['PROTECT_SOUTH', burning.reduce((sum, cell) => sum + cell.row + 1, 0)],
    ['PROTECT_WEST', burning.reduce((sum, cell) => sum + state.gridSize - cell.col, 0)],
    ['PROTECT_EAST', burning.reduce((sum, cell) => sum + cell.col + 1, 0)],
  ];
}

function selectRuleDemoAction(state: SimulationState): string {
  if (state.resources.available <= 0) return 'HOLD';
  const scores = visibleFireSectorScores(state);
  if (scores.every(([, score]) => score === 0)) return 'RECON';
  return scores.reduce((best, candidate) =>
    candidate[1] > best[1] ? candidate : best,
  )[0];
}

function selectDemoAction(state: SimulationState): string {
  if (state.policy === 'random') {
    const sample = hash01(state.stepCount + 1, state.seed ?? 0, 97);
    return ACTIONS[Math.min(ACTIONS.length - 1, Math.floor(sample * ACTIONS.length))].value;
  }
  return selectRuleDemoAction(state);
}

function demoDecisionReason(
  state: SimulationState,
  action: string,
  manuallySelected: boolean,
  resourceBlocked: boolean,
): string {
  const blockedCopy = resourceBlocked
    ? ' No resource was available, so the move had no effect.'
    : '';
  if (manuallySelected) return 'You selected this move for one step.' + blockedCopy;
  if (state.policy === 'random') {
    return (
      'The seeded Random baseline sampled ' +
      actionLabel(action) +
      ' from all six allowed moves.' +
      blockedCopy
    );
  }
  const actor = state.policy.startsWith('ppo_')
    ? 'The Rule Agent fallback'
    : state.policy === 'learning'
      ? 'The offline demo (rule behavior; real Q-learning needs the backend link)'
      : 'The Rule Agent';
  if (action === 'RECON') {
    return actor + ' saw no visible fire readings, so it chose Recon.';
  }
  if (action.startsWith('PROTECT_')) {
    const side = action.replace('PROTECT_', '').toLowerCase();
    return (
      actor +
      ' found the highest visible-fire edge score on the ' +
      side +
      ' side, so it chose ' +
      actionLabel(action) +
      '.'
    );
  }
  return actor + ' chose Hold because no response resources were left.';
}

function detectDemoFailures(
  previous: SimulationState,
  action: string,
  stepCount: number,
  resourceBlocked: boolean,
): AgentFailure[] {
  const trueBurning = previous.cells.filter(
    (cell) => (cell.trueFire ?? cell.fire) === 'burning',
  );
  const hiddenBurning = trueBurning.filter((cell) => !cell.observed).length;
  const context = {
    step: stepCount,
    policy: previous.policy,
    hiddenFireCells: hiddenBurning,
    visibleFireCells: trueBurning.length - hiddenBurning,
    trueBurningCells: trueBurning.length,
    source: 'local_demo' as const,
  };
  const failures: AgentFailure[] = [];
  const record = (failureType: AgentFailureType, message: string) => {
    failures.push({
      id: 'demo-failure-' + stepCount + '-' + failureType,
      failureType,
      message,
      ...context,
    });
  };

  if (trueBurning.length > 0 && hiddenBurning * 2 > trueBurning.length) {
    record(
      'perception_gap',
      'Perception failure: ' + hiddenBurning + ' of ' + trueBurning.length +
        ' burning square(s) were hidden from the agent at decision time.',
    );
  }
  if (action === 'HOLD' && previous.resources.available > 0 && trueBurning.length > 0) {
    record(
      'idle_while_burning',
      'Decision failure: the agent held with resources available while ' +
        trueBurning.length + ' square(s) were truly burning.',
    );
  }
  if (resourceBlocked) {
    record(
      'resource_exhausted_request',
      'Resource failure: ' + actionLabel(action) +
        ' was requested with no response resources remaining.',
    );
  }
  if (action.startsWith('PROTECT_') && trueBurning.length > 0) {
    const half = previous.gridSize / 2;
    const counts: Record<string, number> = {
      NORTH: trueBurning.filter((cell) => cell.row < Math.ceil(half)).length,
      SOUTH: trueBurning.filter((cell) => cell.row >= Math.floor(half)).length,
      EAST: trueBurning.filter((cell) => cell.col >= Math.floor(half)).length,
      WEST: trueBurning.filter((cell) => cell.col < Math.ceil(half)).length,
    };
    const chosen = action.replace('PROTECT_', '');
    const best = Math.max(...Object.values(counts));
    const bestSectors = Object.keys(counts).filter((name) => counts[name] === best);
    if (bestSectors.length === 1 && counts[chosen] < best) {
      record(
        'misdirected_protection',
        'Decision failure: the agent protected the ' + chosen.toLowerCase() +
          ' sector (' + counts[chosen] + ' true burning square(s)) while the ' +
          bestSectors[0].toLowerCase() + ' sector held ' + best + '.',
      );
    }
  }
  return failures;
}

export function advanceDemoState(
  previous: SimulationState,
  requestedAction?: string,
): SimulationState {
  if (previous.done) return previous;
  const stepCount = previous.stepCount + 1;
  const action = normalizeAction(requestedAction ?? selectDemoAction(previous));
  const resourceBlocked = action !== 'HOLD' && previous.resources.available <= 0;
  const resourcesUsed = action !== 'HOLD' && !resourceBlocked ? 1 : 0;
  const protectedRegion = (cell: GridCell): boolean => {
    if (action === 'PROTECT_NORTH') return cell.row < previous.gridSize / 2;
    if (action === 'PROTECT_SOUTH') return cell.row >= previous.gridSize / 2;
    if (action === 'PROTECT_EAST') return cell.col >= previous.gridSize / 2;
    if (action === 'PROTECT_WEST') return cell.col < previous.gridSize / 2;
    return false;
  };
  const activeResourceAction = action.startsWith('PROTECT_') && resourcesUsed === 1;
  const firstPass = previous.cells.map((cell): GridCell => {
    const protection = activeResourceAction && protectedRegion(cell)
      ? Math.max(cell.protection * 0.96, cell.infrastructure ? 0.9 : 0.46)
      : cell.protection * 0.95;
    if ((cell.trueFire ?? cell.fire) === 'burning') {
      const intensity = cell.intensity + 0.12 + hash01(cell.row, cell.col, stepCount) * 0.08;
      return {
        ...cell,
        protection,
        fire: intensity >= 1 ? 'burned' : 'burning',
        trueFire: intensity >= 1 ? 'burned' : 'burning',
        intensity: clamp(intensity),
      };
    }
    return { ...cell, protection };
  });
  const windDirection = previous.wind.direction.toUpperCase();
  const nextCells = firstPass.map((cell): GridCell => {
    if ((cell.trueFire ?? cell.fire) !== 'none' || cell.terrain === 'water') return cell;
    const burningNeighbors = neighborsOf(cell, previous.cells, previous.gridSize).filter(
      (neighbor) => (neighbor.trueFire ?? neighbor.fire) === 'burning',
    );
    if (burningNeighbors.length === 0) return cell;
    const downwind = burningNeighbors.some((neighbor) => {
      const rowDelta = cell.row - neighbor.row;
      const colDelta = cell.col - neighbor.col;
      return (
        (windDirection.includes('N') && rowDelta < 0) ||
        (windDirection.includes('S') && rowDelta > 0) ||
        (windDirection.includes('E') && colDelta > 0) ||
        (windDirection.includes('W') && colDelta < 0)
      );
    });
    const fuelFactor = cell.terrain === 'vegetation' ? 1 : 0.28;
    const spreadChance =
      (0.055 + burningNeighbors.length * 0.055 + (downwind ? 0.09 : 0)) *
      fuelFactor *
      (1 - cell.protection * 0.82);
    return hash01(cell.row, cell.col, stepCount + 31) < spreadChance
      ? { ...cell, fire: 'burning', trueFire: 'burning', intensity: 0.18 }
      : cell;
  });
  const observedCells = nextCells.map((cell) => ({
    ...cell,
    observed:
      action === 'RECON' && resourcesUsed === 1
        ? hash01(cell.row, cell.col, stepCount + 2) < Math.min(1, previous.sensor.reliability + 0.16)
        : hash01(cell.row, cell.col, stepCount + 2) < previous.sensor.reliability,
  }));
  const resources = {
    ...previous.resources,
    available: Math.max(0, previous.resources.available - resourcesUsed),
    consumed: Math.min(
      previous.resources.initial,
      previous.resources.consumed + resourcesUsed,
    ),
  };
  const before = deriveMetrics(previous.cells, previous.metrics);
  const metrics = demoMetrics(
    observedCells,
    previous.metrics.cumulativeReward,
    resources.consumed,
  );
  const newBurning = observedCells.filter(
    (cell, index) =>
      (previous.cells[index].trueFire ?? previous.cells[index].fire) === 'none' &&
      (cell.trueFire ?? cell.fire) === 'burning',
  ).length;
  const newLosses = Math.max(0, metrics.infrastructureLost - before.infrastructureLost);
  const rewardBreakdown = {
    newSpreadPenalty: newBurning > 0 ? newBurning * -0.75 : 0,
    infrastructureLossPenalty: newLosses > 0 ? newLosses * -30 : 0,
    resourceUsePenalty: resourcesUsed > 0 ? resourcesUsed * -0.3 : 0,
    activeFirePenalty:
      metrics.burningCells > 0 ? metrics.burningCells * -0.03 : 0,
    total: 0,
  };
  rewardBreakdown.total =
    rewardBreakdown.newSpreadPenalty +
    rewardBreakdown.infrastructureLossPenalty +
    rewardBreakdown.resourceUsePenalty +
    rewardBreakdown.activeFirePenalty;
  metrics.cumulativeReward =
    previous.metrics.cumulativeReward + rewardBreakdown.total;
  const stepFailures = detectDemoFailures(previous, action, stepCount, resourceBlocked);
  const events = [...previous.events];
  for (const failure of stepFailures) {
    events.push({
      id: 'event-' + failure.id,
      step: stepCount,
      type: 'agent_failure',
      message: failure.message,
      severity: 'warning',
    });
  }
  if (resourceBlocked) {
    events.push({
      id: 'demo-resource-blocked-' + stepCount,
      step: stepCount,
      type: 'resource',
      message: actionLabel(action) + ' was requested, but no response resources were left.',
      severity: 'warning',
    });
  }
  if (newBurning > 0) {
    events.push({
      id: 'demo-spread-' + stepCount,
      step: stepCount,
      type: 'fire',
      message: 'Fire spread into ' + newBurning + ' new map square' + (newBurning === 1 ? '.' : 's.'),
      severity: 'warning',
    });
  } else if (stepCount % 3 === 0) {
    events.push({
      id: 'demo-action-' + stepCount,
      step: stepCount,
      type: 'policy',
      message: actionLabel(action) + ' directive executed by ' + policyLabel(previous.policy) + '.',
      severity: 'info',
    });
  }
  const noActiveFire = metrics.burningCells === 0;
  const done = noActiveFire || stepCount >= (previous.maxSteps ?? 120);
  if (done) {
    events.push({
      id: 'demo-complete-' + stepCount,
      step: stepCount,
      type: 'system',
      message: noActiveFire ? 'No active fire signatures remain.' : 'Simulation terminal condition reached.',
      severity: noActiveFire ? 'success' : 'critical',
    });
  }

  return {
    ...previous,
    stepCount,
    cells: observedCells,
    sensor: {
      ...previous.sensor,
      observedFraction:
        observedCells.filter((cell) => cell.observed).length / observedCells.length,
      reconActive: action === 'RECON' && resourcesUsed === 1,
    },
    resources,
    metrics,
    currentAction: action,
    lastReward: rewardBreakdown.total,
    agentDecision: {
      seenSummary:
        'It had fire readings for ' +
        previous.cells.filter((cell) => cell.observed).length +
        ' map squares, including ' +
        previous.cells.filter(
          (cell) => cell.observed && cell.fire === 'burning',
        ).length +
        ' visible fire squares.',
      visibleCells: previous.cells.filter((cell) => cell.observed).length,
      visibleFireCells: previous.cells.filter(
        (cell) => cell.observed && cell.fire === 'burning',
      ).length,
      hiddenCells: previous.cells.filter((cell) => !cell.observed).length,
      action,
      actionLabel: actionLabel(action),
      decisionSource: requestedAction ? 'manual' : 'policy',
      reason: demoDecisionReason(
        previous,
        action,
        Boolean(requestedAction),
        resourceBlocked,
      ),
      resourcesBeforeAction: previous.resources.available,
      rewardDelta: rewardBreakdown.total,
      rewardBreakdown,
      confidence: null,
      explored: null,
      explorationRate: null,
      source: 'local_demo',
    },
    agentFailures: [...previous.agentFailures, ...stepFailures].slice(-25),
    events: events.slice(-60),
    done,
    terminationReason: done
      ? noActiveFire
        ? 'fire_extinguished'
        : 'max_steps'
      : null,
  };
}

export function injectDemoChaos(previous: SimulationState): SimulationState {
  const eventBase = {
    step: previous.stepCount,
    severity: 'critical' as const,
  };
  const mode = previous.stepCount % 5;
  if (mode === 0) {
    return {
      ...previous,
      wind: {
        direction: 'SW',
        observedDirection: previous.sensor.reliability >= 0.7 ? 'SW' : null,
        strength: 0.92,
      },
      events: [
        ...previous.events,
        {
          ...eventBase,
          id: 'chaos-wind-' + previous.stepCount,
          type: 'weather',
          message: 'CHAOS: abrupt synthetic wind shift detected — front model invalidated.',
        },
      ].slice(-60),
    };
  }
  if (mode === 1) {
    return {
      ...previous,
      sensor: {
        ...previous.sensor,
        reliability: Math.max(0.25, previous.sensor.reliability - 0.25),
        status: 'degraded',
      },
      wind: { ...previous.wind, observedDirection: null },
      events: [
        ...previous.events,
        {
          ...eventBase,
          id: 'chaos-sensor-' + previous.stepCount,
          type: 'sensor',
          message: 'CHAOS: sensor cluster C-4 forced offline.',
        },
      ].slice(-60),
    };
  }
  if (mode === 2) {
    return {
      ...previous,
      sensor: { ...previous.sensor, communicationDegraded: true, status: 'degraded' },
      events: [
        ...previous.events,
        {
          ...eventBase,
          id: 'chaos-comms-' + previous.stepCount,
          type: 'communications',
          message: 'CHAOS: telemetry relay latency exceeds synthetic operating limit.',
        },
      ].slice(-60),
    };
  }
  if (mode === 3) {
    const resources = {
      ...previous.resources,
      available: Math.max(0, previous.resources.available - 1),
      unavailable: Math.min(previous.resources.initial, previous.resources.unavailable + 1),
    };
    resources.consumed = Math.min(
      resources.initial,
      Math.max(0, resources.initial - resources.available - resources.unavailable),
    );
    return {
      ...previous,
      resources,
      metrics: { ...previous.metrics, resourcesConsumed: resources.consumed },
      events: [
        ...previous.events,
        {
          ...eventBase,
          id: 'chaos-resource-' + previous.stepCount,
          type: 'resource',
          message: 'CHAOS: one response resource marked unavailable.',
        },
      ].slice(-60),
    };
  }
  const cells = previous.cells.map((cell) =>
    cell.observed && hash01(cell.row, cell.col, previous.stepCount + 88) > 0.55
      ? { ...cell, observed: false }
      : cell,
  );
  return {
    ...previous,
    cells,
    sensor: {
      ...previous.sensor,
      observedFraction: cells.filter((cell) => cell.observed).length / cells.length,
      status: 'degraded',
    },
    events: [
      ...previous.events,
      {
        ...eventBase,
        id: 'chaos-observation-' + previous.stepCount,
        type: 'sensor',
        message: 'CHAOS: partial observation packet loss injected.',
      },
    ].slice(-60),
  };
}

export function policyLabel(policyId: string): string {
  return DEFAULT_POLICIES.find((policy) => policy.id === policyId)?.label ??
    policyId.replaceAll('_', ' ');
}
