import { useState, useCallback, useRef, useEffect } from 'react';
import { transcribeAudio, fetchSpeechHealth } from '../lib/api';

export type SpeechState = 'idle' | 'recording' | 'transcribing';

interface UseSpeechOptions {
  onTranscript?: (text: string) => void | Promise<void>;
}

export function useSpeech(options: UseSpeechOptions = {}) {
  const [state, setState] = useState<SpeechState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Voice activity detection
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const speechDetectedRef = useRef(false);
  const silenceStartedRef = useRef<number | null>(null);
  const recordingStartedRef = useRef<number>(0);
  const stopRecordingRef = useRef<(() => Promise<string>) | null>(null);

  const onTranscriptRef = useRef(options.onTranscript);

  useEffect(() => {
    onTranscriptRef.current = options.onTranscript;
  }, [options.onTranscript]);

  useEffect(() => {
    fetchSpeechHealth()
      .then((health) => setAvailable(health.available))
      .catch(() => setAvailable(false));
  }, []);

  const cleanupAudioDetection = useCallback(() => {
    if (vadTimerRef.current !== null) {
      window.clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    speechDetectedRef.current = false;
    silenceStartedRef.current = null;
  }, []);

  const startVoiceActivityDetection = useCallback(() => {
    const audioContext = audioContextRef.current;
    const analyser = analyserRef.current;

    if (!audioContext || !analyser) {
      return;
    }

    const data = new Uint8Array(analyser.fftSize);

    // Check microphone level every 50 ms.
    vadTimerRef.current = window.setInterval(() => {
      if (mediaRecorderRef.current?.state !== 'recording') {
        return;
      }

      const now = performance.now();

      // Give the microphone 500 ms to initialize.
      if (now - recordingStartedRef.current < 500) {
        return;
      }

      analyser.getByteTimeDomainData(data);

      let sum = 0;

      for (let i = 0; i < data.length; i += 1) {
        const normalized = (data[i] - 128) / 128;
        sum += normalized * normalized;
      }

      const rms = Math.sqrt(sum / data.length);

      // Voice threshold.
      const isSpeech = rms > 0.025;

      if (isSpeech) {
        speechDetectedRef.current = true;
        silenceStartedRef.current = null;
        return;
      }

      // Don't stop until actual speech has been detected.
      if (!speechDetectedRef.current) {
        return;
      }

      if (silenceStartedRef.current === null) {
        silenceStartedRef.current = now;
        return;
      }

      // 1.2 seconds of silence = end of speech.
      if (now - silenceStartedRef.current >= 700) {
        const recorder = mediaRecorderRef.current;

        if (recorder?.state === 'recording') {
          void stopRecordingRef.current?.();
        }
      }
    }, 50);
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;

      // Set up microphone level monitoring for VAD.
      const AudioContextClass =
        window.AudioContext ||
        (window as typeof window & {
          webkitAudioContext?: typeof AudioContext;
        }).webkitAudioContext;

      if (AudioContextClass) {
        const audioContext = new AudioContextClass();
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();

        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.2;

        source.connect(analyser);

        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
      }

      speechDetectedRef.current = false;
      silenceStartedRef.current = null;
      recordingStartedRef.current = performance.now();

      setState('recording');

      startVoiceActivityDetection();
    } catch (err) {
      console.error('Microphone start failed:', err);
      setError('Microphone access denied');
      setState('idle');
    }
  }, [startVoiceActivityDetection]);

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve, reject) => {
      const recorder = mediaRecorderRef.current;

      if (!recorder || recorder.state !== 'recording') {
        reject(new Error('Not recording'));
        return;
      }

      cleanupAudioDetection();

      recorder.onstop = async () => {
        const transcriptionStart = performance.now();
        setState('transcribing');

        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRecorderRef.current = null;

        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });

        chunksRef.current = [];

        try {
          const result = await transcribeAudio(blob);
          const text = result.text?.trim() || '';

          console.log(
            `[Voice] STT: ${(performance.now() - transcriptionStart).toFixed(0)} ms`,
          );

          setState('idle');

          if (text) {
            console.log(`[Voice] Transcript: "${text}"`);
            await onTranscriptRef.current?.(text);
          }

          resolve(text);
        } catch (err) {
          setState('idle');

          const msg =
            err instanceof Error
              ? err.message
              : 'Transcription failed';

          setError(msg);
          reject(err);
        }
      };

      recorder.stop();
    });
  }, [cleanupAudioDetection]);

  useEffect(() => {
    stopRecordingRef.current = stopRecording;

    return () => {
      stopRecordingRef.current = null;
    };
  }, [stopRecording]);
  // Clean up when the component unmounts.
  useEffect(() => {
    return () => {
      cleanupAudioDetection();

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [cleanupAudioDetection]);

  return {
    state,
    error,
    available,
    startRecording,
    stopRecording,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
  };
}