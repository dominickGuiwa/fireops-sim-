import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement>;

const baseProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

export function FlameIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M12.4 2.6c.8 3.8-2.7 4.6-1.1 7.7.8 1.5 2.1 1.7 2.7.7.7-1.2.1-2.8-.3-3.9 3.1 1.8 5 4.7 4.6 8-.4 3.6-3 6.1-6.3 6.1-3.8 0-6.6-2.8-6.4-6.6.2-3.7 3.1-5.8 6.8-12Z" />
      <path d="M11.7 13.1c1.5 1.7 2.2 2.7 1.9 4-.2 1-1 1.8-2 1.8-1.2 0-2.1-.9-2.1-2.1 0-1.3.9-2.2 2.2-3.7Z" />
    </svg>
  );
}

export function ShieldIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M12 3 4.8 6v5.4c0 4.5 2.9 7.7 7.2 9.6 4.3-1.9 7.2-5.1 7.2-9.6V6L12 3Z" />
      <path d="m8.7 12 2.1 2.1 4.7-4.7" />
    </svg>
  );
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m8 5 11 7-11 7V5Z" />
    </svg>
  );
}

export function PauseIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M8 5v14M16 5v14" />
    </svg>
  );
}

export function StepIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m6 5 10 7-10 7V5ZM19 5v14" />
    </svg>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M4.9 7.4A8.5 8.5 0 1 1 3.7 14" />
      <path d="M3.5 5v4.7h4.7" />
    </svg>
  );
}

export function ChaosIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m13.4 2.8-8.7 11h6.4l-.8 7.4 9-11.7h-6.2l.3-6.7Z" />
    </svg>
  );
}

export function RadioIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="12" cy="12" r="1.5" />
      <path d="M8.6 8.7a4.7 4.7 0 0 0 0 6.6M15.4 8.7a4.7 4.7 0 0 1 0 6.6M5.7 5.8a8.8 8.8 0 0 0 0 12.4M18.3 5.8a8.8 8.8 0 0 1 0 12.4" />
    </svg>
  );
}

export function ActivityIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M3 12h4l2-6 4 12 2-6h6" />
    </svg>
  );
}

export function ResourceIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M4 7.5 12 3l8 4.5v9L12 21l-8-4.5v-9Z" />
      <path d="m4.3 7.7 7.7 4.5 7.7-4.5M12 12.2V21" />
    </svg>
  );
}

export function TargetIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
    </svg>
  );
}

export function ChevronIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m8 10 4 4 4-4" />
    </svg>
  );
}

export function AlertIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M12 3 2.8 20h18.4L12 3Z" />
      <path d="M12 9v5M12 17.4v.1" />
    </svg>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M20 7v5h-5M4 17v-5h5" />
      <path d="M6.1 8.1A7.5 7.5 0 0 1 18.8 10M17.9 15.9A7.5 7.5 0 0 1 5.2 14" />
    </svg>
  );
}

export function MedicalIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M9 4h6v5h5v6h-5v5H9v-5H4V9h5V4Z" />
    </svg>
  );
}

export function PowerIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m13.4 2.8-8.7 11h6.4l-.8 7.4 9-11.7h-6.2l.3-6.7Z" />
    </svg>
  );
}

export function CommunicationsIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M12 10v11M8 21h8M10 14l-2 7M14 14l2 7" />
      <circle cx="12" cy="7" r="2" />
      <path d="M7.8 2.8a6 6 0 0 0 0 8.4M16.2 2.8a6 6 0 0 1 0 8.4" />
    </svg>
  );
}

export function HelpIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.7 9a2.4 2.4 0 1 1 3.2 2.3c-.9.4-1.3 1-1.3 2M12 17.2v.1" />
    </svg>
  );
}

export function EyeIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M2.8 12s3.3-5.5 9.2-5.5 9.2 5.5 9.2 5.5-3.3 5.5-9.2 5.5S2.8 12 2.8 12Z" />
      <circle cx="12" cy="12" r="2.4" />
    </svg>
  );
}

export function BrainIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M9.3 5.2A3 3 0 0 1 15 6.5a3.2 3.2 0 0 1 2.7 4.8 3.3 3.3 0 0 1-1.6 5.8 3 3 0 0 1-5 2.2V4.9M9.3 5.2a3 3 0 0 0-4.5 3.3 3.2 3.2 0 0 0 1.5 5.9 3.3 3.3 0 0 0 4.8 4.9" />
      <path d="M7.3 9.3c1.3 0 2.1.7 2.1 1.8M16.4 13.6c-1.1 0-1.8.6-1.8 1.6" />
    </svg>
  );
}

export function BookIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M4 4.5h5.2A2.8 2.8 0 0 1 12 7.3V20a3.5 3.5 0 0 0-3.5-3.5H4v-12ZM20 4.5h-5.2A2.8 2.8 0 0 0 12 7.3V20a3.5 3.5 0 0 1 3.5-3.5H20v-12Z" />
    </svg>
  );
}

export function ArrowRightIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M5 12h14M14 7l5 5-5 5" />
    </svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m6 6 12 12M18 6 6 18" />
    </svg>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <svg {...baseProps} {...props}>
      <path d="m12 2 1.5 5.2L18 9l-4.5 1.8L12 16l-1.5-5.2L6 9l4.5-1.8L12 2ZM19 15l.7 2.3L22 18l-2.3.7L19 21l-.7-2.3L16 18l2.3-.7L19 15ZM5 3l.6 1.9L7.5 5l-1.9.6L5 7.5l-.6-1.9L2.5 5l1.9-.6L5 3Z" />
    </svg>
  );
}
