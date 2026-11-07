interface JarvisVisualizerProps {
  speaking: boolean;
  processing?: boolean;
  deepProcessing?: boolean;
  intensiveProcessing?: boolean;
  status?: string;
}

export function JarvisVisualizer({
  speaking,
  processing = false,
  deepProcessing = false,
  status,
}: JarvisVisualizerProps) {
  const mode = speaking
    ? 'speaking'
    : deepProcessing
      ? 'deep-processing'
      : processing
        ? 'processing'
        : 'idle';

  const label =
    status ||
    (speaking
      ? 'J.A.R.V.I.S • SPEAKING'
      : deepProcessing
        ? 'DEEP PROCESSING'
        : processing
          ? 'PROCESSING'
          : 'J.A.R.V.I.S');

  return (
    <div
      className={`jarvis-visualizer jarvis-mode-${mode}`}
      aria-hidden="true"
    >
      {/* Ambient energy */}
      <div className="jarvis-energy jarvis-energy-1" />
      <div className="jarvis-energy jarvis-energy-2" />
      <div className="jarvis-energy jarvis-energy-3" />

      {/* Rotating HUD rings */}
      <div className="jarvis-orbit jarvis-orbit-1" />
      <div className="jarvis-orbit jarvis-orbit-2" />
      <div className="jarvis-orbit jarvis-orbit-3" />
      <div className="jarvis-orbit jarvis-orbit-4" />

      {/* Orbital markers */}
      <div className="jarvis-marker jarvis-marker-1" />
      <div className="jarvis-marker jarvis-marker-2" />
      <div className="jarvis-marker jarvis-marker-3" />
      <div className="jarvis-marker jarvis-marker-4" />

      {/* Scanning beam */}
      <div className="jarvis-scan-line" />

      {/* Neural connections */}
      <div className="jarvis-neural-network">
        <span />
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>

      {/* Expanding response waves */}
      <div className="jarvis-wave jarvis-wave-1" />
      <div className="jarvis-wave jarvis-wave-2" />
      <div className="jarvis-wave jarvis-wave-3" />

      {/* Central core */}
      <div className="jarvis-core">
        <div className="jarvis-core-halo" />
        <div className="jarvis-core-glow" />

        <div className="jarvis-core-inner">
          <div className="jarvis-core-dots">
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>

      {/* HUD status */}
      <div className="jarvis-visualizer-label">
        <span className="jarvis-status-dot" />
        {label}
      </div>
    </div>
  );
}