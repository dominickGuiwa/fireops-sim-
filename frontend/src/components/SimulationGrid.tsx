import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
  CommunicationsIcon,
  FlameIcon,
  MedicalIcon,
  PowerIcon,
  ShieldIcon,
} from './Icons';
import type {
  ConnectionStatus,
  GridCell,
  InfrastructureType,
  SimulationState,
} from '../types/simulation';

type ViewMode = 'observation' | 'truth';

interface SimulationGridProps {
  state: SimulationState;
  connection: ConnectionStatus;
}

const directionDegrees: Record<string, number> = {
  N: 0,
  NORTH: 0,
  NNE: 22.5,
  NE: 45,
  ENE: 67.5,
  E: 90,
  EAST: 90,
  ESE: 112.5,
  SE: 135,
  SSE: 157.5,
  S: 180,
  SOUTH: 180,
  SSW: 202.5,
  SW: 225,
  WSW: 247.5,
  W: 270,
  WEST: 270,
  WNW: 292.5,
  NW: 315,
  NNW: 337.5,
};

function infrastructureLabel(type: InfrastructureType | null): string {
  if (type === 'medical') return 'medical facility';
  if (type === 'power') return 'power facility';
  if (type === 'communications') return 'communications facility';
  return '';
}

function cellLabel(
  cell: GridCell,
  masked: boolean,
  displayFire: GridCell['fire'],
  showProtection: boolean,
): string {
  const coordinate =
    'row ' + String(cell.row + 1).padStart(2, '0') + ', column ' + String(cell.col + 1).padStart(2, '0');
  const details = [
    cell.terrain,
    infrastructureLabel(cell.infrastructure),
    masked ? 'fire sensor data unavailable' : displayFire !== 'none' ? displayFire : '',
    showProtection ? 'protected' : '',
  ].filter(Boolean);
  return coordinate + ': ' + details.join(', ');
}

function InfrastructureMarker({ type }: { type: InfrastructureType }) {
  return (
    <span className={'infrastructure-marker infrastructure-marker--' + type}>
      {type === 'medical' ? <MedicalIcon /> : null}
      {type === 'power' ? <PowerIcon /> : null}
      {type === 'communications' ? <CommunicationsIcon /> : null}
    </span>
  );
}

function WindCompass({
  direction,
  strength,
  mode,
}: {
  direction: string | null;
  strength: number | null;
  mode: ViewMode;
}) {
  const normalizedDirection = direction?.toUpperCase() ?? 'UNKNOWN';
  const signalLost = direction === null || directionDegrees[normalizedDirection] === undefined;
  const degrees = directionDegrees[normalizedDirection] ?? 0;
  const force =
    strength === null
      ? null
      : Math.round(Math.min(1, strength > 1 ? strength / 100 : strength) * 100);
  const label =
    mode === 'truth'
      ? signalLost
        ? 'True wind direction unavailable'
        : 'True wind ' + normalizedDirection + ', force ' + force + ' percent'
      : signalLost
        ? 'Observed wind direction unavailable. Wind strength is not given to the agent.'
        : 'Observed wind ' +
          normalizedDirection +
          '. Wind strength is not given to the agent.';

  return (
    <div
      className={'wind-compass ' + (signalLost ? 'wind-compass--lost' : '')}
      aria-label={label}
    >
      <div className="wind-compass__dial">
        <span className="wind-compass__north">N</span>
        <svg viewBox="0 0 96 96" aria-hidden="true">
          <circle cx="48" cy="48" r="35" />
          <circle cx="48" cy="48" r="23" className="wind-compass__inner" />
          {signalLost ? (
            <path className="wind-compass__lost-mark" d="m36 36 24 24m0-24L36 60" />
          ) : (
            <g transform={'rotate(' + degrees + ' 48 48)'}>
              <path className="wind-compass__arrow" d="M48 14 55 49 48 44 41 49 48 14Z" />
              <path className="wind-compass__tail" d="m48 82-4-30 4 3 4-3-4 30Z" />
            </g>
          )}
          <circle cx="48" cy="48" r="3" className="wind-compass__hub" />
        </svg>
      </div>
      <div className="wind-compass__copy">
        <span>{mode === 'truth' ? 'True wind vector' : 'Observed wind'}</span>
        <strong>{normalizedDirection}</strong>
        <small>
          {signalLost ? 'SIGNAL LOST' : mode === 'truth' ? 'FORCE ' + force + '%' : 'FORCE HIDDEN'}
        </small>
      </div>
    </div>
  );
}

function MapLegend() {
  const items = [
    { className: 'legend-swatch--vegetation', label: 'Vegetation' },
    { className: 'legend-swatch--burning', label: 'Active fire' },
    { className: 'legend-swatch--burned', label: 'Burned' },
    { className: 'legend-swatch--road', label: 'Road / open' },
    { className: 'legend-swatch--masked', label: 'Fire data gap' },
  ];

  return (
    <div className="map-legend" aria-label="Map legend">
      {items.map((item) => (
        <span className="map-legend__item" key={item.label}>
          <i className={'legend-swatch ' + item.className} />
          {item.label}
        </span>
      ))}
      <span className="map-legend__divider" />
      <span className="map-legend__item">
        <MedicalIcon className="legend-icon legend-icon--medical" />
        Medical
      </span>
      <span className="map-legend__item">
        <PowerIcon className="legend-icon legend-icon--power" />
        Power
      </span>
      <span className="map-legend__item">
        <CommunicationsIcon className="legend-icon legend-icon--communications" />
        Comms
      </span>
    </div>
  );
}

export function SimulationGrid({ state, connection }: SimulationGridProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('observation');
  const [hoveredCell, setHoveredCell] = useState<GridCell | null>(null);
  const [selectedCoordinate, setSelectedCoordinate] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const axisLabels = useMemo(
    () =>
      Array.from({ length: state.gridSize }, (_, index) =>
        index === 0 || (index + 1) % 5 === 0 ? String(index + 1).padStart(2, '0') : '',
      ),
    [state.gridSize],
  );
  const truthAvailable = state.cells.some((cell) => cell.trueFire !== null);
  useEffect(() => {
    if (!truthAvailable && viewMode === 'truth') {
      setViewMode('observation');
    }
  }, [truthAvailable, viewMode]);
  const selectedCell = selectedCoordinate
    ? state.cells.find(
        (cell) =>
          cell.row === selectedCoordinate.row && cell.col === selectedCoordinate.col,
      ) ?? null
    : null;
  const inspectedCell = hoveredCell ?? selectedCell;
  const inspectedMasked = Boolean(
    inspectedCell && viewMode === 'observation' && !inspectedCell.observed,
  );
  const inspectedFire =
    inspectedMasked
      ? 'none'
      : inspectedCell && viewMode === 'truth' && inspectedCell.trueFire !== null
      ? inspectedCell.trueFire
      : inspectedCell?.fire ?? 'none';
  const inspectedProtection = Boolean(
    inspectedCell && viewMode === 'truth' && inspectedCell.protection > 0.1,
  );
  const inspectedCopy = inspectedCell
    ? cellLabel(inspectedCell, inspectedMasked, inspectedFire, inspectedProtection)
    : 'Hover, tap, or use arrow keys to inspect a cell';

  const moveGridSelection = (rowDelta: number, colDelta: number) => {
    const current = selectedCoordinate ?? { row: 0, col: 0 };
    setSelectedCoordinate({
      row: Math.max(0, Math.min(state.gridSize - 1, current.row + rowDelta)),
      col: Math.max(0, Math.min(state.gridSize - 1, current.col + colDelta)),
    });
  };

  return (
    <section className="map-panel panel">
      <div className="map-panel__header">
        <div>
          <span className="section-kicker">Synthetic practice area</span>
          <h1>WILDFIRE MAP</h1>
          <p>Watch the synthetic fire and the three sites the agent must protect.</p>
        </div>
        <div className="view-control">
          <div className="view-selector" role="group" aria-label="Map view">
            <button
              type="button"
              className={viewMode === 'observation' ? 'is-active' : ''}
              onClick={() => setViewMode('observation')}
              aria-pressed={viewMode === 'observation'}
            >
              Agent view
            </button>
            <button
              type="button"
              className={viewMode === 'truth' ? 'is-active' : ''}
              onClick={() => setViewMode('truth')}
              disabled={!truthAvailable}
              aria-pressed={viewMode === 'truth'}
            >
              {truthAvailable ? 'Truth view' : 'Truth unavailable'}
            </button>
          </div>
          <span className="view-selector__help">
            Truth is for you. The agent cannot see it.
          </span>
        </div>
      </div>

      <div className="map-stage">
        <div className="map-axis map-axis--x" aria-hidden="true">
          {axisLabels.map((label, index) => (
            <span key={index}>{label}</span>
          ))}
        </div>
        <div className="map-axis map-axis--y" aria-hidden="true">
          {axisLabels.map((label, index) => (
            <span key={index}>{label}</span>
          ))}
        </div>
        <div className="grid-corner-label" aria-hidden="true">
          Y/X
        </div>
        <div className="grid-frame">
          <div
            className="simulation-grid"
            role="grid"
            tabIndex={0}
            aria-label={'Synthetic wildfire grid, ' + state.gridSize + ' by ' + state.gridSize}
            aria-activedescendant={
              selectedCoordinate
                ? 'grid-cell-' + selectedCoordinate.row + '-' + selectedCoordinate.col
                : undefined
            }
            style={{
              gridTemplateColumns: 'repeat(' + state.gridSize + ', minmax(0, 1fr))',
            }}
            onPointerLeave={() => setHoveredCell(null)}
            onFocus={() =>
              setSelectedCoordinate((current) => current ?? { row: 0, col: 0 })
            }
            onKeyDown={(event) => {
              if (event.key === 'ArrowUp') moveGridSelection(-1, 0);
              else if (event.key === 'ArrowDown') moveGridSelection(1, 0);
              else if (event.key === 'ArrowLeft') moveGridSelection(0, -1);
              else if (event.key === 'ArrowRight') moveGridSelection(0, 1);
              else if (event.key === 'Home') setSelectedCoordinate({ row: 0, col: 0 });
              else if (event.key === 'End') {
                setSelectedCoordinate({ row: state.gridSize - 1, col: state.gridSize - 1 });
              } else return;
              event.preventDefault();
            }}
          >
            {state.cells.map((cell) => {
              const masked = viewMode === 'observation' && !cell.observed;
              const displayFire =
                masked
                  ? 'none'
                  : viewMode === 'truth' && cell.trueFire !== null
                    ? cell.trueFire
                    : cell.fire;
              const showProtection = viewMode === 'truth' && cell.protection > 0.1;
              const style = {
                '--fire-intensity': String(masked ? 0 : cell.intensity),
                '--cell-delay': String(((cell.row * 3 + cell.col) % 8) * -0.12) + 's',
              } as CSSProperties;
              const classNames = [
                'sim-cell',
                'sim-cell--' + cell.terrain,
                displayFire !== 'none' ? 'sim-cell--' + displayFire : '',
                masked ? 'sim-cell--masked' : '',
                showProtection ? 'sim-cell--protected' : '',
                cell.infrastructure ? 'sim-cell--infrastructure' : '',
                selectedCoordinate?.row === cell.row && selectedCoordinate.col === cell.col
                  ? 'sim-cell--selected'
                  : '',
              ]
                .filter(Boolean)
                .join(' ');
              return (
                <div
                  key={cell.row + '-' + cell.col}
                  id={'grid-cell-' + cell.row + '-' + cell.col}
                  className={classNames}
                  role="gridcell"
                  aria-label={cellLabel(cell, masked, displayFire, showProtection)}
                  title={cellLabel(cell, masked, displayFire, showProtection)}
                  style={style}
                  onPointerEnter={() => setHoveredCell(cell)}
                  onClick={() => setSelectedCoordinate({ row: cell.row, col: cell.col })}
                >
                  {displayFire === 'burning' ? (
                    <span className="fire-glyph" aria-hidden="true">
                      <FlameIcon />
                      <i />
                    </span>
                  ) : null}
                  {cell.infrastructure ? <InfrastructureMarker type={cell.infrastructure} /> : null}
                  {showProtection ? (
                    <ShieldIcon className="protection-marker" aria-hidden="true" />
                  ) : null}
                  {!cell.observed ? <span className="sensor-gap-dot" aria-hidden="true" /> : null}
                </div>
              );
            })}
          </div>
          <div className="grid-scanline" aria-hidden="true" />
          {connection === 'offline' ? (
            <span className="demo-watermark">LOCAL DEMO FEED</span>
          ) : null}
        </div>
        <WindCompass
          direction={viewMode === 'truth' ? state.wind.direction : state.wind.observedDirection}
          strength={viewMode === 'truth' ? state.wind.strength : null}
          mode={viewMode}
        />
      </div>

      <div className="map-panel__footer">
        <MapLegend />
        <div className="cell-inspector" aria-live="polite">
          <span className="cell-inspector__reticle">⌖</span>
          <span>{inspectedCopy}</span>
        </div>
      </div>
    </section>
  );
}
