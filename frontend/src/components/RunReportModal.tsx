import type { GuidedRunReport } from '../lib/guidance';
import { outcomeLabel } from '../lib/guidance';
import { AccessibleModal } from './AccessibleModal';
import {
  ArrowRightIcon,
  BookIcon,
  FlameIcon,
  ResetIcon,
  ShieldIcon,
  SparkIcon,
} from './Icons';

interface RunReportModalProps {
  open: boolean;
  report: GuidedRunReport | null;
  onClose: () => void;
  onPlayAgain: () => void;
}

function signed(value: number | null): string {
  if (value === null) return '—';
  const formatted = value.toFixed(1);
  return value > 0 ? '+' + formatted : formatted;
}

export function RunReportModal({
  open,
  report,
  onClose,
  onPlayAgain,
}: RunReportModalProps) {
  if (!report) return null;
  return (
    <AccessibleModal
      open={open}
      title={report.title}
      eyebrow="End-of-run report"
      description={
        outcomeLabel(report.outcome) +
        ' · ' +
        (report.source === 'server summary'
          ? 'Summary supplied by the simulator'
          : 'Summary calculated from the final results')
      }
      className={'report-modal report-modal--' + report.outcome}
      onClose={onClose}
      footer={
        <>
          <button className="modal-button modal-button--quiet" type="button" onClick={onClose}>
            Close report
          </button>
          <button className="modal-button modal-button--primary" type="button" onClick={onPlayAgain}>
            <ResetIcon />
            Play again
          </button>
        </>
      }
    >
      <div className="report-outcome">
        <div className="report-outcome__mark">
          {report.outcome === 'success' ? <ShieldIcon /> : <FlameIcon />}
        </div>
        <div>
          <span>Result</span>
          <strong>{outcomeLabel(report.outcome)}</strong>
          <p>
            {report.terminationReason === 'fire_extinguished'
              ? 'The fire went out.'
              : 'The simulation reached its step limit.'}
          </p>
        </div>
        <div className="report-xp">
          <SparkIcon />
          <span>Practice XP</span>
          <strong>+{report.xpEarned}</strong>
        </div>
      </div>

      <div className="report-score-grid">
        <div>
          <span>Sites safe</span>
          <strong>{report.infrastructurePreserved}/{report.infrastructureTotal}</strong>
        </div>
        <div>
          <span>Fire-affected squares</span>
          <strong>{report.burnedArea}/{report.totalCells}</strong>
        </div>
        <div>
          <span>Resources used</span>
          <strong>{report.resourcesConsumed}</strong>
        </div>
        <div>
          <span>Total reward</span>
          <strong>{signed(report.totalReward)}</strong>
        </div>
      </div>

      <div className="report-checks" aria-label="Mission and success checks">
        <div>
          <span>Fire went out?</span>
          <strong>{report.missionCompleted ? 'Yes' : 'No'}</strong>
          <small>Yes means the fire went out before the step limit.</small>
        </div>
        <div>
          <span>Full mission success?</span>
          <strong>{report.successful ? 'Yes' : 'No'}</strong>
          <small>Full success means the fire went out and all 3 synthetic sites stayed safe.</small>
        </div>
      </div>

      <div className="report-sections">
        <section>
          <h3>What happened</h3>
          <ul>
            {report.whatHappened.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
        <section className="report-lesson">
          <h3><SparkIcon /> Reward lesson</h3>
          <p>{report.rewardLesson}</p>
          <div>
            Last step: <strong>{signed(report.stepReward)}</strong>
            <ArrowRightIcon />
            Trial total: <strong>{signed(report.totalReward)}</strong>
          </div>
        </section>
        <section>
          <h3>Try this next</h3>
          <ul>
            {report.improvements.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      </div>

      <div className="report-disclaimer">
        <BookIcon />
        <p>
          <strong>Synthetic learning environment.</strong>
          This report is not for real emergency, evacuation, firefighting, or life-safety
          decisions.
        </p>
      </div>
    </AccessibleModal>
  );
}
