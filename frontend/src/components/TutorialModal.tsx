import { useEffect, useState } from 'react';
import { AccessibleModal } from './AccessibleModal';
import {
  ArrowRightIcon,
  BookIcon,
  BrainIcon,
  ChaosIcon,
  EyeIcon,
  PlayIcon,
  ShieldIcon,
  SparkIcon,
  StepIcon,
} from './Icons';

interface TutorialModalProps {
  open: boolean;
  onClose: () => void;
}

const slides = [
  {
    title: 'Your mission',
    text: 'Keep all 3 synthetic sites safe and help the fire go out.',
    points: [
      'Medical, power, and communications sites appear on the map.',
      'Fire can spread one step at a time.',
      'The run ends when the fire goes out or the step limit is reached.',
    ],
    icon: 'shield',
  },
  {
    title: 'Two ways to see the map',
    text: 'The two map buttons answer different questions.',
    points: [
      'Agent View shows only what the fixed policy receives.',
      'A sensor gap hides fire data; terrain and sites remain visible.',
      'Truth Overlay is for you, the researcher. The agent cannot see it.',
    ],
    icon: 'eye',
  },
  {
    title: 'Pick the challenge',
    text: 'Choose a policy and decide how reliable the sensors should be.',
    points: [
      'A policy is a fixed method for choosing actions.',
      'Lower reliability hides more information.',
      'PPO options may use a clearly labeled Rule Agent fallback when no model is installed.',
    ],
    icon: 'brain',
  },
  {
    title: 'Control the run',
    text: 'You can move at your own pace.',
    points: [
      'Run keeps taking steps. Pause stops after the current step.',
      'Step moves forward once. Reset starts a new trial.',
      'Inject Chaos adds one synthetic problem, such as a sensor or wind failure.',
    ],
    icon: 'controls',
  },
  {
    title: 'Read the score',
    text: 'The simulator explains each move and what it changed.',
    points: [
      'A better step loses fewer points. Total reward adds every step penalty.',
      'The end report says what happened and what to try next.',
      'XP tracks practice in this browser. It does not train the agent.',
    ],
    icon: 'spark',
  },
  {
    title: 'Ready to experiment',
    text: 'Try a run, change one setting, and compare what happens.',
    points: [
      'There is no single perfect score for every broken-sensor case.',
      'Help opens this guide again at any time.',
      'This synthetic simulator is not for real emergency or life-safety decisions.',
    ],
    icon: 'book',
  },
] as const;

function SlideIcon({ icon }: { icon: (typeof slides)[number]['icon'] }) {
  if (icon === 'shield') return <ShieldIcon />;
  if (icon === 'eye') return <EyeIcon />;
  if (icon === 'brain') return <BrainIcon />;
  if (icon === 'controls') {
    return (
      <span className="tutorial-control-icons">
        <PlayIcon />
        <StepIcon />
        <ChaosIcon />
      </span>
    );
  }
  if (icon === 'spark') return <SparkIcon />;
  return <BookIcon />;
}

export function TutorialModal({ open, onClose }: TutorialModalProps) {
  const [step, setStep] = useState(0);
  const slide = slides[step];

  useEffect(() => {
    if (open) setStep(0);
  }, [open]);

  const isLast = step === slides.length - 1;

  return (
    <AccessibleModal
      open={open}
      title="How FIREOPS-SIM works"
      eyebrow="Quick guide"
      description="Six short cards. Use Next and Back, Skip at any time, or press Escape to close."
      className="tutorial-modal"
      onClose={onClose}
      footer={
        <>
          <button className="modal-button modal-button--quiet" type="button" onClick={onClose}>
            Skip
          </button>
          <div className="tutorial-progress" aria-label={'Step ' + (step + 1) + ' of ' + slides.length}>
            {slides.map((item, index) => (
              <span className={index === step ? 'is-active' : ''} key={item.title} />
            ))}
          </div>
          <div className="modal-actions">
            <button
              className="modal-button modal-button--secondary"
              type="button"
              onClick={() => setStep((current) => Math.max(0, current - 1))}
              disabled={step === 0}
            >
              Back
            </button>
            <button
              className="modal-button modal-button--primary"
              type="button"
              onClick={() => {
                if (isLast) onClose();
                else setStep((current) => Math.min(slides.length - 1, current + 1));
              }}
            >
              {isLast ? 'Start exploring' : 'Next'}
              {!isLast ? <ArrowRightIcon /> : null}
            </button>
          </div>
        </>
      }
    >
      <div className="tutorial-slide">
        <div className="tutorial-slide__icon">
          <SlideIcon icon={slide.icon} />
        </div>
        <span className="tutorial-slide__count">
          {String(step + 1).padStart(2, '0')} / {String(slides.length).padStart(2, '0')}
        </span>
        <h3>{slide.title}</h3>
        <p>{slide.text}</p>
        <ul>
          {slide.points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      </div>
    </AccessibleModal>
  );
}
