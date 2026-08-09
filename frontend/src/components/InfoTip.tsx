import { useId } from 'react';

interface InfoTipProps {
  term: string;
  children: string;
}

export function InfoTip({ term, children }: InfoTipProps) {
  const tooltipId = useId();
  return (
    <span className="info-tip">
      <button
        type="button"
        className="info-tip__button"
        aria-label={'What does ' + term + ' mean?'}
        aria-describedby={tooltipId}
      >
        ?
      </button>
      <span className="info-tip__bubble" id={tooltipId} role="tooltip">
        <strong>{term}</strong>
        {children}
      </span>
    </span>
  );
}
