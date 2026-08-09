import type { SimulationEvent } from '../types/simulation';
import { AlertIcon } from './Icons';

interface EventLogProps {
  events: SimulationEvent[];
}

export function EventLog({ events }: EventLogProps) {
  const visibleEvents = [...events].reverse().slice(0, 8);

  return (
    <section className="panel event-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Step-by-step record</span>
          <h2>WHAT JUST HAPPENED</h2>
        </div>
        <span className="event-count">{events.length} EVENTS</span>
      </div>

      <div className="event-list" aria-live="polite" aria-label="Simulation event log">
        {visibleEvents.length === 0 ? (
          <div className="event-empty">No events reported for this episode.</div>
        ) : (
          visibleEvents.map((event) => (
            <article className={'event-item event-item--' + event.severity} key={event.id}>
              <div className="event-item__rail">
                <span />
                <i />
              </div>
              <div className="event-item__body">
                <div className="event-item__meta">
                  <span>T+{String(event.step).padStart(3, '0')}</span>
                  <strong>{event.type.toUpperCase()}</strong>
                </div>
                <p>{event.message}</p>
              </div>
              {event.severity === 'critical' ? <AlertIcon /> : null}
            </article>
          ))
        )}
      </div>
    </section>
  );
}
