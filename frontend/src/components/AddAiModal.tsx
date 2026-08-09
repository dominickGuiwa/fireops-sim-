import { useCallback, useState } from 'react';
import { registerPolicyPrototype } from '../api/client';
import type { ConnectionStatus } from '../types/simulation';
import { AccessibleModal } from './AccessibleModal';

interface AddAiModalProps {
  open: boolean;
  connection: ConnectionStatus;
  onClose: () => void;
}

export function AddAiModal({ open, connection, onClose }: AddAiModalProps) {
  const [serverReply, setServerReply] = useState<string | null>(null);
  const [isPinging, setIsPinging] = useState(false);

  const pingEndpoint = useCallback(async () => {
    setIsPinging(true);
    try {
      setServerReply(await registerPolicyPrototype());
    } catch (error) {
      setServerReply(
        error instanceof Error ? error.message : 'The registration endpoint could not be reached.',
      );
    } finally {
      setIsPinging(false);
    }
  }, []);

  return (
    <AccessibleModal
      open={open}
      onClose={onClose}
      eyebrow="Prototype"
      title="ADD A NEW AI"
      description="A placeholder for future policy plugins. Pressing anything here never interrupts the running simulation."
      className="add-ai-modal"
      footer={
        <button type="button" className="modal-button" onClick={onClose}>
          Back to the simulator
        </button>
      }
    >
      <div className="add-ai-body">
        <p>
          New agents plug into the simulator through one deliberately small
          contract, so any future AI — scripted, tabular, or deep — competes
          under identical rules:
        </p>
        <ol className="add-ai-steps">
          <li>
            <strong>Implement the Agent protocol.</strong> One method,
            <code> act(observation) → action 0-5</code>, receiving only the
            degraded sensor observation. No agent ever sees the truth grid.
          </li>
          <li>
            <strong>Register it in the session.</strong> Today that is one line in
            <code> backend/app/services/session.py</code>; saved PPO or DQN
            artifacts load lazily through environment variables.
          </li>
          <li>
            <strong>Prove it under failure.</strong> The benchmark replays the
            same seeded worlds, chaos schedules, and reliability levels for every
            policy, and the failure analyzer flags its mistakes.
          </li>
        </ol>
        <p className="add-ai-note">
          A runtime registration API is planned; until it ships, the endpoint
          below honestly answers <code>501 Not Implemented</code> and the
          simulator keeps running.
        </p>
        <button
          type="button"
          className="training-button"
          disabled={isPinging || connection !== 'online'}
          onClick={() => void pingEndpoint()}
        >
          {isPinging ? 'Contacting…' : 'Ping POST /api/policies/register'}
        </button>
        {connection !== 'online' ? (
          <p className="add-ai-reply">The backend link is offline, so the live 501 reply cannot be shown.</p>
        ) : serverReply ? (
          <p className="add-ai-reply" role="status">
            <strong>Server replied:</strong> {serverReply}
          </p>
        ) : null}
      </div>
    </AccessibleModal>
  );
}
