import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getPolicies,
  getSimulationState,
  injectChaos as requestChaos,
  resetSimulation,
  stepSimulation,
  trainLearningPolicy,
  updateSimulationConfig,
} from '../api/client';
import {
  advanceDemoState,
  clamp,
  createDemoState,
  injectDemoChaos,
  normalizePolicies,
  normalizeSimulationState,
} from '../lib/simulation';
import {
  DEFAULT_POLICIES,
  type ConnectionStatus,
  type PolicyDefinition,
  type SimulationState,
  type TrainingDifficulty,
} from '../types/simulation';

export interface SimulationController {
  state: SimulationState;
  policies: PolicyDefinition[];
  connection: ConnectionStatus;
  isRunning: boolean;
  isBusy: boolean;
  isTraining: boolean;
  lastError: string | null;
  run: () => void;
  pause: () => void;
  step: (action?: string) => Promise<void>;
  reset: () => Promise<void>;
  injectChaos: () => Promise<void>;
  selectPolicy: (policy: string) => Promise<void>;
  setReliability: (reliability: number) => void;
  train: (episodes: number, difficulty: TrainingDifficulty) => Promise<void>;
  refresh: () => Promise<void>;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'The simulation API could not be reached.';
}

function withReliabilityEvent(
  state: SimulationState,
  reliability: number,
  started: boolean,
): SimulationState {
  const percent = Math.round(reliability * 100);
  return {
    ...state,
    events: [
      ...state.events.filter((event) => !event.id.startsWith('ui-reliability-')),
      {
        id: 'ui-reliability-' + (started ? 'started' : 'pending'),
        step: state.stepCount,
        type: 'settings',
        message: started
          ? 'Map reliability is now ' + percent + '%. A new trial started.'
          : 'Map reliability is changing to ' + percent + '%. The simulation paused and a new trial will start.',
        severity: 'info' as const,
      },
    ].slice(-60),
  };
}

function withPolicyEvent(state: SimulationState): SimulationState {
  return {
    ...state,
    events: [
      ...state.events,
      {
        id: 'ui-policy-started',
        step: state.stepCount,
        type: 'settings',
        message: 'The agent policy changed. A new trial started.',
        severity: 'info' as const,
      },
    ].slice(-60),
  };
}

export function useSimulation(): SimulationController {
  const [state, setState] = useState<SimulationState>(() =>
    createDemoState(Date.now() % 2_147_483_647),
  );
  const [policies, setPolicies] = useState<PolicyDefinition[]>(DEFAULT_POLICIES);
  const [connection, setConnection] = useState<ConnectionStatus>('connecting');
  const [isRunning, setIsRunning] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const stateRef = useRef(state);
  const connectionRef = useRef(connection);
  const commandInFlight = useRef(false);
  const reliabilityTimer = useRef<number | null>(null);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const setConnectionStatus = useCallback((status: ConnectionStatus) => {
    connectionRef.current = status;
    setConnection(status);
  }, []);

  const applyPayload = useCallback((payload: unknown) => {
    setState((current) => normalizeSimulationState(payload, current));
  }, []);

  const markOffline = useCallback(
    (error: unknown) => {
      setConnectionStatus('offline');
      setLastError(errorMessage(error));
    },
    [setConnectionStatus],
  );

  const refresh = useCallback(async () => {
    try {
      const payload = await getSimulationState();
      applyPayload(payload);
      setConnectionStatus('online');
      setLastError(null);
    } catch (error) {
      markOffline(error);
    }
  }, [applyPayload, markOffline, setConnectionStatus]);

  useEffect(() => {
    void refresh();
    void getPolicies()
      .then((payload) => setPolicies(normalizePolicies(payload)))
      .catch(() => {
        // Built-in policies keep the controls useful while the API is unavailable.
      });
  }, [refresh]);

  useEffect(() => {
    const retryDelay = connection === 'offline' ? 8_000 : 15_000;
    const retry = window.setInterval(() => {
      if (!commandInFlight.current && !isRunning) {
        void refresh();
      }
    }, retryDelay);
    return () => window.clearInterval(retry);
  }, [connection, isRunning, refresh]);

  const step = useCallback(
    async (action?: string) => {
      if (commandInFlight.current || stateRef.current.done) return;
      commandInFlight.current = true;
      setIsBusy(true);
      try {
        if (connectionRef.current === 'online') {
          const payload = await stepSimulation(action);
          applyPayload(payload);
          setLastError(null);
        } else {
          setState((current) => advanceDemoState(current, action));
        }
      } catch (error) {
        markOffline(error);
        setState((current) => advanceDemoState(current, action));
      } finally {
        commandInFlight.current = false;
        setIsBusy(false);
      }
    },
    [applyPayload, markOffline],
  );

  useEffect(() => {
    if (!isRunning || state.done) return undefined;
    const ticker = window.setInterval(() => void step(), 850);
    return () => window.clearInterval(ticker);
  }, [isRunning, state.done, step]);

  useEffect(() => {
    if (state.done && isRunning) {
      setIsRunning(false);
    }
  }, [isRunning, state.done]);

  useEffect(
    () => () => {
      if (reliabilityTimer.current !== null) {
        window.clearTimeout(reliabilityTimer.current);
      }
    },
    [],
  );

  const cancelPendingReliabilityUpdate = useCallback(() => {
    if (reliabilityTimer.current !== null) {
      window.clearTimeout(reliabilityTimer.current);
      reliabilityTimer.current = null;
    }
  }, []);

  const reset = useCallback(async () => {
    if (commandInFlight.current) return;
    cancelPendingReliabilityUpdate();
    setIsRunning(false);
    commandInFlight.current = true;
    setIsBusy(true);
    const current = stateRef.current;
    try {
      if (connectionRef.current === 'online') {
        const payload = await resetSimulation({
          policy: current.policy,
          sensorReliability: current.sensor.configuredReliability,
        });
        applyPayload(payload);
        setLastError(null);
      } else {
        setState(
          createDemoState(
            Date.now() % 2_147_483_647,
            current.policy,
            current.sensor.configuredReliability,
          ),
        );
      }
    } catch (error) {
      markOffline(error);
      setState(
        createDemoState(
          Date.now() % 2_147_483_647,
          current.policy,
          current.sensor.configuredReliability,
        ),
      );
    } finally {
      commandInFlight.current = false;
      setIsBusy(false);
    }
  }, [applyPayload, cancelPendingReliabilityUpdate, markOffline]);

  const injectChaos = useCallback(async () => {
    if (commandInFlight.current) return;
    commandInFlight.current = true;
    setIsBusy(true);
    try {
      if (connectionRef.current === 'online') {
        const payload = await requestChaos();
        applyPayload(payload);
        setLastError(null);
      } else {
        setState(injectDemoChaos);
      }
    } catch (error) {
      markOffline(error);
      setState(injectDemoChaos);
    } finally {
      commandInFlight.current = false;
      setIsBusy(false);
    }
  }, [applyPayload, markOffline]);

  const selectPolicy = useCallback(
    async (policy: string) => {
      cancelPendingReliabilityUpdate();
      setIsRunning(false);
      const reliability = stateRef.current.sensor.configuredReliability;
      if (connectionRef.current !== 'online') {
        setState(
          withPolicyEvent(
            createDemoState(Date.now() % 2_147_483_647, policy, reliability),
          ),
        );
        return;
      }
      try {
        const payload = await resetSimulation({
          policy,
          sensorReliability: reliability,
        });
        applyPayload(payload);
        setState(withPolicyEvent);
        setLastError(null);
      } catch (error) {
        markOffline(error);
        setState(
          withPolicyEvent(
            createDemoState(Date.now() % 2_147_483_647, policy, reliability),
          ),
        );
      }
    },
    [applyPayload, cancelPendingReliabilityUpdate, markOffline],
  );

  const train = useCallback(
    async (episodes: number, difficulty: TrainingDifficulty) => {
      if (commandInFlight.current || connectionRef.current !== 'online') return;
      commandInFlight.current = true;
      setIsRunning(false);
      setIsTraining(true);
      setIsBusy(true);
      try {
        await trainLearningPolicy(episodes, difficulty);
        const payload = await getSimulationState();
        applyPayload(payload);
        setLastError(null);
      } catch (error) {
        markOffline(error);
      } finally {
        commandInFlight.current = false;
        setIsTraining(false);
        setIsBusy(false);
      }
    },
    [applyPayload, markOffline],
  );

  const setReliability = useCallback(
    (value: number) => {
      const reliability = clamp(value);
      setIsRunning(false);
      setState((current) =>
        withReliabilityEvent(
          {
            ...current,
            wind: {
              ...current.wind,
              observedDirection:
                reliability < 0.35
                  ? null
                  : current.wind.observedDirection ?? current.wind.direction,
            },
            sensor: {
              ...current.sensor,
              reliability,
              configuredReliability: reliability,
              status:
                reliability < 0.4
                  ? 'offline'
                  : reliability < 0.78
                    ? 'degraded'
                    : 'online',
            },
          },
          reliability,
          false,
        ),
      );
      if (reliabilityTimer.current !== null) {
        window.clearTimeout(reliabilityTimer.current);
      }
      reliabilityTimer.current = window.setTimeout(() => {
        reliabilityTimer.current = null;
        if (connectionRef.current !== 'online') {
          setState(
            withReliabilityEvent(
              createDemoState(
                Date.now() % 2_147_483_647,
                stateRef.current.policy,
                reliability,
              ),
              reliability,
              true,
            ),
          );
          return;
        }
        void updateSimulationConfig({ sensorReliability: reliability })
          .then((payload) => {
            applyPayload(payload);
            setState((current) => withReliabilityEvent(current, reliability, true));
            setLastError(null);
          })
          .catch((error) => {
            markOffline(error);
            setState(
              withReliabilityEvent(
                createDemoState(
                  Date.now() % 2_147_483_647,
                  stateRef.current.policy,
                  reliability,
                ),
                reliability,
                true,
              ),
            );
          });
      }, 320);
    },
    [applyPayload, markOffline],
  );

  return {
    state,
    policies,
    connection,
    isRunning,
    isBusy,
    isTraining,
    lastError,
    run: () => setIsRunning(true),
    pause: () => setIsRunning(false),
    step,
    reset,
    injectChaos,
    selectPolicy,
    setReliability,
    train,
    refresh,
  };
}
