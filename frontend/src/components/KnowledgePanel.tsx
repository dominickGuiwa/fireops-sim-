import { useCallback, useState, type FormEvent } from 'react';
import { searchKnowledge } from '../api/client';
import type { ConnectionStatus, KnowledgeAnswer, KnowledgeResult } from '../types/simulation';
import { BookIcon } from './Icons';
import { InfoTip } from './InfoTip';

interface KnowledgePanelProps {
  connection: ConnectionStatus;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeKnowledge(payload: unknown): KnowledgeAnswer | null {
  if (!isRecord(payload) || typeof payload.query !== 'string') return null;
  const results: KnowledgeResult[] = Array.isArray(payload.results)
    ? payload.results
        .map((raw): KnowledgeResult | null => {
          if (!isRecord(raw)) return null;
          if (typeof raw.title !== 'string' || typeof raw.text !== 'string') return null;
          return {
            rank: typeof raw.rank === 'number' ? raw.rank : 0,
            title: raw.title,
            text: raw.text,
            score: typeof raw.score === 'number' ? raw.score : 0,
          };
        })
        .filter((item): item is KnowledgeResult => item !== null)
    : [];
  return {
    query: payload.query,
    results,
    answer: typeof payload.answer === 'string' ? payload.answer : null,
    note: typeof payload.note === 'string' ? payload.note : null,
  };
}

export function KnowledgePanel({ connection }: KnowledgePanelProps) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState<KnowledgeAnswer | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const offline = connection !== 'online';

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = query.trim();
      if (trimmed.length < 2 || offline || isSearching) return;
      setIsSearching(true);
      setError(null);
      try {
        setAnswer(normalizeKnowledge(await searchKnowledge(trimmed)));
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : 'The field manual could not be searched.',
        );
      } finally {
        setIsSearching(false);
      }
    },
    [isSearching, offline, query],
  );

  return (
    <section className="panel knowledge-panel" aria-labelledby="knowledge-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Retrieval-augmented help</span>
          <h2 id="knowledge-title">FIELD MANUAL (RAG)</h2>
        </div>
        <span className="panel-code">
          <InfoTip term="Field manual">
            Your question is matched against the synthetic simulator manual with
            TF-IDF retrieval. Answers are quoted from the cited section — no
            generative model, and never real emergency advice.
          </InfoTip>
        </span>
      </div>

      <form className="knowledge-form" onSubmit={(event) => void submit(event)}>
        <input
          type="text"
          value={query}
          minLength={2}
          maxLength={200}
          placeholder="e.g. when should the agent use recon?"
          aria-label="Search the simulator field manual"
          disabled={offline}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="submit" disabled={offline || isSearching || query.trim().length < 2}>
          <BookIcon />
          {isSearching ? 'Searching…' : 'Search'}
        </button>
      </form>

      {offline ? (
        <p className="knowledge-empty">
          The field manual index lives on the backend. Reconnect the API link to search it.
        </p>
      ) : error ? (
        <p className="knowledge-empty">{error}</p>
      ) : answer === null ? (
        <p className="knowledge-empty">
          Ask about actions, sensors, chaos, training difficulty, or the failure
          panel. Every answer is quoted from a cited manual section.
        </p>
      ) : answer.results.length === 0 ? (
        <p className="knowledge-empty">
          No manual section matched “{answer.query}”. Try simulator terms like
          “recon”, “ember jump”, or “reward”.
        </p>
      ) : (
        <div className="knowledge-results">
          {answer.answer ? (
            <blockquote className="knowledge-answer">
              <p>{answer.answer}</p>
              <cite>— {answer.results[0].title}</cite>
            </blockquote>
          ) : null}
          <ul>
            {answer.results.map((result) => (
              <li key={result.title}>
                <div className="knowledge-result-head">
                  <strong>{result.title}</strong>
                  <span>match {(result.score * 100).toFixed(0)}%</span>
                </div>
                <p>{result.text}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
