import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE_URL } from './api/client';
import { AddAiModal } from './components/AddAiModal';
import { AgentDecisionCard } from './components/AgentDecisionCard';
import { AlertIcon, RefreshIcon } from './components/Icons';
import { ControlBar } from './components/ControlBar';
import { EventLog } from './components/EventLog';
import { FailurePanel } from './components/FailurePanel';
import { Header } from './components/Header';
import { KnowledgePanel } from './components/KnowledgePanel';
import { MetricsPanel } from './components/MetricsPanel';
import { MissionBrief } from './components/MissionBrief';
import { PolicyPanel } from './components/PolicyPanel';
import { RunReportModal } from './components/RunReportModal';
import { SimulationGrid } from './components/SimulationGrid';
import { SystemStatus } from './components/SystemStatus';
import { TrainingPanel } from './components/TrainingPanel';
import { TrialNotebook } from './components/TrialNotebook';
import { TutorialModal } from './components/TutorialModal';
import { useSimulation } from './hooks/useSimulation';
import { buildRunReport, type GuidedRunReport } from './lib/guidance';
import {
  clearPracticeLog,
  loadPracticeLog,
  markTutorialSeen,
  savePracticeTrial,
  tutorialWasSeen,
  type PracticeLog,
} from './lib/trialLog';

type ActiveModal = 'tutorial' | 'report' | 'add-ai' | null;

function App() {
  const simulation = useSimulation();
  const [activeModal, setActiveModal] = useState<ActiveModal>(() =>
    tutorialWasSeen() ? null : 'tutorial',
  );
  const [latestReport, setLatestReport] = useState<GuidedRunReport | null>(null);
  const [practiceLog, setPracticeLog] = useState<PracticeLog>(() => loadPracticeLog());
  const handledReportIds = useRef<Set<string> | null>(null);
  if (handledReportIds.current === null) {
    handledReportIds.current = new Set(practiceLog.knownIds);
  }
  const selectedPolicy = useMemo(
    () => simulation.policies.find((policy) => policy.id === simulation.state.policy),
    [simulation.policies, simulation.state.policy],
  );

  const closeTutorial = useCallback(() => {
    markTutorialSeen();
    setActiveModal(simulation.state.done && latestReport ? 'report' : null);
  }, [latestReport, simulation.state.done]);

  const openTutorial = useCallback(() => {
    simulation.pause();
    setActiveModal('tutorial');
  }, [simulation]);

  const closeReport = useCallback(() => setActiveModal(null), []);

  const resetTrial = useCallback(() => {
    setActiveModal(null);
    void simulation.reset();
  }, [simulation]);

  const clearPracticeData = useCallback(() => {
    setPracticeLog(clearPracticeLog());
    setLatestReport(null);
    if (simulation.state.done) {
      void simulation.reset();
    }
  }, [simulation]);

  useEffect(() => {
    if (!simulation.state.done) return;
    const report = buildRunReport(simulation.state);
    setLatestReport(report);
    if (handledReportIds.current?.has(report.id)) return;
    handledReportIds.current?.add(report.id);
    setPracticeLog((current) => savePracticeTrial(report, current).log);
    setActiveModal((current) => (current === 'tutorial' ? current : 'report'));
  }, [simulation.state]);

  return (
    <div className="app-shell">
      <div className="ambient-grid" aria-hidden="true" />
      <Header
        connection={simulation.connection}
        step={simulation.state.stepCount}
        isRunning={simulation.isRunning}
        onRefresh={() => void simulation.refresh()}
        onHelp={openTutorial}
      />

      {simulation.connection === 'offline' ? (
        <div className="connection-banner" role="status">
          <AlertIcon />
          <div>
            <strong>Backend link unavailable</strong>
            <span>
              Interactive local demo feed active. API target: {API_BASE_URL}
              {simulation.lastError ? ' · ' + simulation.lastError : ''}
            </span>
          </div>
          <button type="button" onClick={() => void simulation.refresh()}>
            <RefreshIcon />
            Retry link
          </button>
        </div>
      ) : null}

      <main className="command-layout">
        <div className="primary-column">
          <MissionBrief state={simulation.state} />
          <SimulationGrid state={simulation.state} connection={simulation.connection} />
          <div className="learning-grid">
            <AgentDecisionCard state={simulation.state} policy={selectedPolicy} />
            <TrialNotebook
              log={practiceLog}
              serverStats={simulation.state.trialStats}
              canViewLatest={latestReport !== null}
              onViewLatest={() => setActiveModal('report')}
              onClear={clearPracticeData}
            />
          </div>
          <FailurePanel state={simulation.state} />
          <KnowledgePanel connection={simulation.connection} />
        </div>
        <aside className="right-rail" aria-label="Simulation controls and telemetry">
          <PolicyPanel
            state={simulation.state}
            policies={simulation.policies}
            onSelectPolicy={(policy) => void simulation.selectPolicy(policy)}
            onReliabilityChange={simulation.setReliability}
            onAddAi={() => setActiveModal('add-ai')}
          />
          <TrainingPanel
            state={simulation.state}
            connection={simulation.connection}
            isTraining={simulation.isTraining}
            isBusy={simulation.isBusy}
            onTrain={(episodes, difficulty) => void simulation.train(episodes, difficulty)}
          />
          <SystemStatus state={simulation.state} />
          <MetricsPanel state={simulation.state} />
          <EventLog events={simulation.state.events} />
        </aside>
      </main>

      <ControlBar
        isRunning={simulation.isRunning}
        isBusy={simulation.isBusy}
        done={simulation.state.done}
        stepCount={simulation.state.stepCount}
        onRun={simulation.run}
        onPause={simulation.pause}
        onStep={(action) => void simulation.step(action)}
        onReset={resetTrial}
        onChaos={() => void simulation.injectChaos()}
      />

      <div className="safety-note">
        <span>RESEARCH USE ONLY</span>
        Simplified synthetic environment · not for real emergency, evacuation, firefighting, or life-safety decisions
      </div>

      <TutorialModal open={activeModal === 'tutorial'} onClose={closeTutorial} />
      <AddAiModal
        open={activeModal === 'add-ai'}
        connection={simulation.connection}
        onClose={() => setActiveModal(null)}
      />
      <RunReportModal
        open={activeModal === 'report'}
        report={latestReport}
        onClose={closeReport}
        onPlayAgain={resetTrial}
      />
    </div>
  );
}

export default App;
