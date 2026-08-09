import type { ResetOptions } from '../types/simulation';

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(
  /\/+$/,
  '',
);
const REQUEST_TIMEOUT_MS = 4_000;

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(API_BASE_URL + path, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errorPayload = (await response.json()) as { detail?: unknown };
        if (typeof errorPayload.detail === 'string') {
          detail = errorPayload.detail;
        }
      } catch {
        // The response may not contain JSON.
      }
      throw new ApiError('API request failed: ' + detail, response.status);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('API request timed out');
    }
    throw new ApiError(error instanceof Error ? error.message : 'API connection failed');
  } finally {
    window.clearTimeout(timeout);
  }
}

export function getSimulationState(): Promise<unknown> {
  return request('/api/state');
}

export function resetSimulation(options: ResetOptions): Promise<unknown> {
  const body = {
    seed: options.seed,
    policy: options.policy,
    config: {
      sensor_reliability: options.sensorReliability,
    },
  };
  return request('/api/reset', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function stepSimulation(action?: string): Promise<unknown> {
  return request('/api/step', {
    method: 'POST',
    body: JSON.stringify(action ? { action } : {}),
  });
}

export function injectChaos(): Promise<unknown> {
  return request('/api/chaos', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function updateSimulationConfig(config: {
  sensorReliability?: number;
}): Promise<unknown> {
  return request('/api/config', {
    method: 'PUT',
    body: JSON.stringify({
      sensor_reliability: config.sensorReliability,
    }),
  });
}

export function getPolicies(): Promise<unknown> {
  return request('/api/policies');
}

export function getLearningStats(): Promise<unknown> {
  return request('/api/learning');
}

// Training runs real episodes synchronously on the server, so allow more time.
export function trainLearningPolicy(
  episodes: number,
  difficulty: string = 'standard',
): Promise<unknown> {
  return request(
    '/api/train',
    {
      method: 'POST',
      body: JSON.stringify({ episodes, difficulty }),
    },
    30_000,
  );
}

export function searchKnowledge(query: string, k = 3): Promise<unknown> {
  const params = new URLSearchParams({ query, k: String(k) });
  return request('/api/knowledge?' + params.toString());
}

// The registration endpoint intentionally answers 501 while the plugin loader
// is a prototype; callers surface that detail instead of treating it as fatal.
export async function registerPolicyPrototype(): Promise<string> {
  try {
    await request('/api/policies/register', { method: 'POST' });
    return 'Unexpected success: the prototype endpoint should answer 501.';
  } catch (error) {
    if (error instanceof ApiError && error.status === 501) {
      return error.message.replace(/^API request failed: /, '');
    }
    throw error;
  }
}

export { API_BASE_URL };
