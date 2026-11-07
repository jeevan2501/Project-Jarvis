import { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Square, Paperclip, Search } from 'lucide-react';
import { toast } from 'sonner';
import { useAppStore, generateId } from '../../lib/store';
import { streamChat, streamResearch } from '../../lib/sse';
import { fetchSavings, synthesizeSpeech } from '../../lib/api';
import { listConnectors, getSyncStatus } from '../../lib/connectors-api';
import { serializeToolCallArguments } from '../../lib/tool-call';
import {
  engineFromCompletionChunk,
  resolveChatEngine,
} from '../../lib/chat-telemetry';
import { MicButton } from './MicButton';
import { JarvisVisualizer } from './JarvisVisualizer';
import { useSpeech } from '../../hooks/useSpeech';
import type {
  ChatMessage,
  MessageTelemetry,
  ResearchSearchTrace,
  ResearchSource,
  TokenUsage,
  ToolCallInfo,
} from '../../types';

// While Deep Research is toggled on, poll connected sources for sync
// progress so we can surface "Searching over N items — sync in progress"
// next to the toggle. Polling is gated on `enabled` so toggling DR off
// stops the network chatter immediately.
function formatToolName(tool: string): string {
  const labels: Record<string, string> = {
    computer_diagnostics: 'Computer Diagnostics',
    process_inspect: 'Process Inspection',
    service_inspect: 'Service Inspection',
    excel_create: 'Creating Excel Spreadsheet',
    docx_create: 'Creating Word Document',
    folder_create: 'Creating Folder',
    web_search: 'Searching the Web',
    file_read: 'Reading File',
    shell_exec: 'Executing Command',
    code_interpreter: 'Running Code',
  };

  return (
    labels[tool] ??
    tool
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase())
  );
}

function useResearchCorpusSync(enabled: boolean): {
  syncing: boolean;
  itemsSynced: number;
} {
  const [state, setState] = useState({ syncing: false, itemsSynced: 0 });

  useEffect(() => {
    if (!enabled) {
      setState({ syncing: false, itemsSynced: 0 });
      return;
    }
    let cancelled = false;

    const poll = async () => {
      try {
        const list = await listConnectors();
        const connected = list.filter((c) => c.connected);
        if (connected.length === 0) {
          if (!cancelled) setState({ syncing: false, itemsSynced: 0 });
          return;
        }
        const results = await Promise.all(
          connected.map(async (c) => {
            try {
              return await getSyncStatus(c.connector_id);
            } catch {
              return null;
            }
          }),
        );
        let syncing = false;
        let itemsSynced = 0;
        for (const r of results) {
          if (!r) continue;
          if (r.state === 'syncing') syncing = true;
          itemsSynced += r.items_synced ?? 0;
        }
        if (!cancelled) setState({ syncing, itemsSynced });
      } catch {
        // Network blip — leave previous state intact.
      }
    };

    poll();
    const interval = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [enabled]);

  return state;
}

export function InputArea() {
  const [input, setInput] = useState('');
  const [ttsBusy, setTtsBusy] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const ttsBufferRef = useRef('');
  const ttsQueueRef = useRef<string[]>([]);
  const ttsAudioQueueRef = useRef<Blob[]>([]);
  const ttsPlayingRef = useRef(false);
  const ttsSynthesizingRef = useRef(false);
  const fillTTSAudioQueueRef = useRef<(() => void) | null>(null);
  const activeId = useAppStore((s) => s.activeId);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const streamState = useAppStore((s) => s.streamState);
  const messages = useAppStore((s) => s.messages);
  const speechEnabled = useAppStore((s) => s.settings.speechEnabled);
  const ttsEnabled = useAppStore((s) => s.settings.ttsEnabled);
  const voiceMode = useAppStore((s) => s.settings.voiceMode);

  const ttsEnabledRef = useRef(ttsEnabled);

  useEffect(() => {
    ttsEnabledRef.current = ttsEnabled;
  }, [ttsEnabled]);
  const maxTokens = useAppStore((s) => s.settings.maxTokens);
  const temperature = useAppStore((s) => s.settings.temperature);
  const createConversation = useAppStore((s) => s.createConversation);
  const addMessage = useAppStore((s) => s.addMessage);
  const updateLastAssistant = useAppStore((s) => s.updateLastAssistant);
  const setStreamState = useAppStore((s) => s.setStreamState);
  const resetStream = useAppStore((s) => s.resetStream);
  const modelLoading = useAppStore((s) => s.modelLoading);
  const deepResearch = useAppStore((s) => s.deepResearch);
  const setDeepResearch = useAppStore((s) => s.setDeepResearch);
  const corpusSync = useResearchCorpusSync(deepResearch);
  const isCurrentChatStreaming = streamState.isStreaming && streamState.conversationId === activeId;

  const {
    state: speechState,
    error: speechError,
    available: speechAvailable,
    startRecording,
    stopRecording,
  } = useSpeech({
    onTranscript: async (text) => {
      if (voiceMode && text.trim()) {
        await sendMessage(text);
      }
    },
  });

  // Abort in-flight stream when the user switches models mid-generation.
  // This prevents errors from trying to continue a stream with a stale model.
  const prevModelRef = useRef(selectedModel);
  useEffect(() => {
    if (prevModelRef.current !== selectedModel && streamState.isStreaming) {
      abortRef.current?.abort();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      resetStream();
      abortRef.current = null;
    }
    prevModelRef.current = selectedModel;
  }, [selectedModel, streamState.isStreaming, resetStream]);

  const micDisabled = !speechEnabled || !speechAvailable || streamState.isStreaming;
  const micReason: 'not-enabled' | 'no-backend' | 'streaming' | undefined =
    !speechEnabled ? 'not-enabled'
    : !speechAvailable ? 'no-backend'
    : streamState.isStreaming ? 'streaming'
    : undefined;
    // Voice Mode: automatically start listening when JARVIS is ready.
  useEffect(() => {
    if (
      voiceMode &&
      speechEnabled &&
      speechAvailable &&
      !streamState.isStreaming &&
      !ttsBusy &&
      speechState === 'idle'
    ) {
      void startRecording();
    }
  }, [
    voiceMode,
    speechEnabled,
    speechAvailable,
    streamState.isStreaming,
    ttsBusy,
    speechState,
    startRecording,
  ]);
  useEffect(() => {
    if (speechError) {
      toast.error(speechError, { duration: 8000 });
    }
  }, [speechError]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [input]);

  const prepareTTSText = useCallback((text: string): string => {
    let clean = text;
    // Remove fenced code blocks.
    clean = clean.replace(/```[\s\S]*?```/g, ' ');
    // Remove inline code markers.
    clean = clean.replace(/`([^`]*)`/g, '$1');
    // Convert common Markdown emphasis to plain text.
    clean = clean.replace(/(\*\*|__)(.*?)\1/g, '$2');
    clean = clean.replace(/[*_~]/g, '');
    // Remove Markdown links but keep their visible text.
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    // Remove LaTeX display/inline delimiters.
    clean = clean.replace(/\\\[|\\\]/g, ' ');
    clean = clean.replace(/\\\(|\\\)/g, ' ');
    clean = clean.replace(/\$\$[\s\S]*?\$\$/g, ' ');
    clean = clean.replace(/\$([^$]+)\$/g, ' ');
    // Remove common LaTeX commands while preserving useful text.
    clean = clean.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '$1 divided by $2');
    clean = clean.replace(/\\sqrt\s*\{([^{}]*)\}/g, 'square root of $1');
    clean = clean.replace(/\\text\s*\{([^{}]*)\}/g, '$1');
    clean = clean.replace(/\\mathrm\s*\{([^{}]*)\}/g, '$1');
    clean = clean.replace(/\\mathbf\s*\{([^{}]*)\}/g, '$1');
    // Remove remaining LaTeX commands.
    clean = clean.replace(/\\[a-zA-Z]+/g, ' ');
    // Remove LaTeX grouping characters and common math symbols.
    clean = clean.replace(/[{}[\]^_]/g, ' ');
    clean = clean.replace(/\\+/g, ' ');
    // Remove Markdown heading/list prefixes.
    clean = clean.replace(/^\s*#{1,6}\s+/gm, '');
    clean = clean.replace(/^\s*[-*+]\s+/gm, '');
    // Normalize whitespace.
    clean = clean.replace(/\s+/g, ' ').trim();
    return clean;
  }, []);
  const playNextTTS = useCallback(async () => {
    if (ttsPlayingRef.current) return;

    const audioBlob = ttsAudioQueueRef.current.shift();
    fillTTSAudioQueueRef.current?.();
    if (!audioBlob) return;

    ttsPlayingRef.current = true;
    setTtsBusy(true);

    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    const cleanup = () => {
      URL.revokeObjectURL(audioUrl);
      ttsPlayingRef.current = false;

      void playNextTTS().then(() => {
        if (
          ttsQueueRef.current.length === 0 &&
          ttsAudioQueueRef.current.length === 0 &&
          !ttsSynthesizingRef.current &&
          !ttsPlayingRef.current
        ) {
          setTtsBusy(false);
        }
      });
    };

    audio.onended = cleanup;
    audio.onerror = cleanup;

    try {
      await audio.play();
    } catch (err) {
      console.warn('TTS playback unavailable:', err);
      cleanup();
    }
  }, []);

  const fillTTSAudioQueue = useCallback(async () => {
    // Keep at most two synthesized sentences ready.
    if (ttsSynthesizingRef.current) return;
    if (ttsQueueRef.current.length === 0) return;
    if (ttsAudioQueueRef.current.length >= 2) return;

    ttsSynthesizingRef.current = true;

    try {
      while (
        ttsQueueRef.current.length > 0 &&
        ttsAudioQueueRef.current.length < 2
      ) {
        const text = ttsQueueRef.current.shift();
        if (!text) continue;

        try {
          const audioBlob = await synthesizeSpeech(
            text,
            'kokoro',
            'am_michael',
            1.0,
          );

          ttsAudioQueueRef.current.push(audioBlob);
          void playNextTTS();
        } catch (err) {
          console.warn('TTS synthesis unavailable:', err);
        }
      }
    } finally {
      ttsSynthesizingRef.current = false;

      // If more text arrived while synthesis was running,
      // continue filling the audio queue.
      if (
        ttsQueueRef.current.length > 0 &&
        ttsAudioQueueRef.current.length < 2
      ) {
        void fillTTSAudioQueue();
      }
    }
  }, [playNextTTS]);

  useEffect(() => {
    fillTTSAudioQueueRef.current = () => {
      void fillTTSAudioQueue();
    };

    return () => {
      fillTTSAudioQueueRef.current = null;
    };
  }, [fillTTSAudioQueue]);

  const enqueueTTS = useCallback(
    (text: string) => {
      if (!ttsEnabledRef.current) return;

      const cleanText = prepareTTSText(text);

      // Avoid speaking tiny fragments.
      if (!cleanText) return;

      ttsQueueRef.current.push(cleanText);
      setTtsBusy(true);
      void fillTTSAudioQueue();
    },
    [fillTTSAudioQueue, prepareTTSText],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();

    ttsQueueRef.current = [];
    ttsAudioQueueRef.current = [];
    ttsBufferRef.current = '';

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    resetStream();
  }, [resetStream]);

  const sendMessage = useCallback(async (voiceText?: string) => {
    const content = (voiceText ?? input).trim();
    if (!content || streamState.isStreaming) return;
    if (!selectedModel) {
      toast.error('Pick a model first (⌘K)');
      return;
    }

    setInput('');

    let convId = activeId;
    if (!convId) {
      convId = createConversation(selectedModel);
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };
    addMessage(convId, userMsg);

    // Build API messages before adding assistant placeholder
    const currentMessages = useAppStore.getState().messages;
    const apiMessages = currentMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isResearch: deepResearch || undefined,
    };
    addMessage(convId, assistantMsg);

    // Start streaming
    const startTime = Date.now();
    const timer = setInterval(() => {
      setStreamState({ elapsedMs: Date.now() - startTime });
    }, 100);
    timerRef.current = timer;

    const controller = new AbortController();
    abortRef.current = controller;

    let accumulatedContent = '';
    let usage: TokenUsage | undefined;
    let complexity: { score: number; tier: string; suggested_max_tokens: number } | undefined;
    let routedEngine: string | undefined;
    const toolCalls: ToolCallInfo[] = [];
    const researchTraces: ResearchSearchTrace[] = [];
    const researchSourcesByRef = new Map<number, ResearchSource>();
    const flushSources = () =>
      Array.from(researchSourcesByRef.values()).sort((a, b) => a.ref - b.ref);
    let lastFlush = 0;
    let ttftMs: number | undefined;
    let generationAborted = false;

    ttsBufferRef.current = '';
    ttsQueueRef.current = [];

    setStreamState({
      conversationId: convId,
      isStreaming: true,
      phase: deepResearch ? 'Researching...' : 'Generating...',
      elapsedMs: 0,
      activeToolCalls: [],
      content: '',
    });
    useAppStore.getState().addLogEntry({
      timestamp: Date.now(),
      level: 'info',
      category: 'chat',
      message: deepResearch
        ? `Research: "${content.slice(0, 80)}${content.length > 80 ? '...' : ''}"`
        : `Request: "${content.slice(0, 80)}${content.length > 80 ? '...' : ''}" → ${selectedModel}`,
    });

    try {
      if (deepResearch) {
        for await (const ev of streamResearch(
          content,
          selectedModel,
          controller.signal,
        )) {
          if (ev.type === 'search_call') {
            const trace: ResearchSearchTrace = {
              id: generateId(),
              query: ev.arguments?.query ?? '',
              person: ev.arguments?.person,
              timeRange: ev.arguments?.time_range,
              status: 'pending',
            };
            researchTraces.push(trace);
            setStreamState({ phase: `Searching: ${trace.query}` });
            updateLastAssistant(
              convId,
              accumulatedContent,
              undefined,
              undefined,
              undefined,
              undefined,
              [...researchTraces],
              flushSources(),
            );
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(),
              level: 'info',
              category: 'tool',
              message: `Search: "${trace.query}"${trace.person ? ` (person: ${trace.person})` : ''}`,
            });
          } else if (ev.type === 'search_result') {
            const pending = [...researchTraces].reverse().find((t) => t.status === 'pending');
            if (pending) {
              pending.status = 'complete';
              pending.numHits = ev.num_hits;
              pending.topTitles = ev.top_titles;
            }
            if (ev.sources) {
              for (const src of ev.sources) {
                if (src && typeof src.ref === 'number' && !researchSourcesByRef.has(src.ref)) {
                  researchSourcesByRef.set(src.ref, src);
                }
              }
            }
            updateLastAssistant(
              convId,
              accumulatedContent,
              undefined,
              undefined,
              undefined,
              undefined,
              [...researchTraces],
              flushSources(),
            );
          } else if (ev.type === 'synthesis') {
            if (!ttftMs) ttftMs = Date.now() - startTime;
            accumulatedContent += ev.text;
            setStreamState({ content: accumulatedContent, phase: '' });
            const now = Date.now();
            if (now - lastFlush >= 80) {
              updateLastAssistant(
                convId,
                accumulatedContent,
                undefined,
                undefined,
                undefined,
                undefined,
                [...researchTraces],
                flushSources(),
              );
              lastFlush = now;
            }
          } else if (ev.type === 'system_metrics') {
            // Live GPU sample — feed straight to the System panel so Power
            // (W) and Energy (kJ) tick up in real time as the agent runs.
            useAppStore.getState().setLiveEnergy({
              power_w: ev.power_w,
              energy_j: ev.energy_j,
              duration_s: ev.duration_s,
            });
          } else if (ev.type === 'error') {
            // Backend setup/worker failure (Ollama down, planner model
            // missing, KnowledgeStore locked, etc.). Without surfacing the
            // message, the user sees only the generic "No response was
            // generated" fallback and has no way to self-diagnose.
            const msg = ev.message || 'Research failed (no detail provided)';
            accumulatedContent = accumulatedContent
              ? `${accumulatedContent}\n\n**Research stopped:** ${msg}`
              : `**Research failed:** ${msg}`;
            setStreamState({ content: accumulatedContent, phase: '' });
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(),
              level: 'error',
              category: 'chat',
              message: `Deep Research error: ${msg}`,
            });
            toast.error(msg, { duration: 8000 });
          } else if (ev.type === 'done') {
            if (ev.usage) {
              usage = {
                prompt_tokens: ev.usage.prompt_tokens ?? 0,
                completion_tokens: ev.usage.completion_tokens ?? 0,
                total_tokens:
                  ev.usage.total_tokens ??
                  (ev.usage.prompt_tokens ?? 0) +
                    (ev.usage.completion_tokens ?? 0),
              };
              // Optimistically roll this research turn into the session
              // counters so the Session panel updates the moment the
              // stream finishes, regardless of how /v1/savings aggregates
              // research telemetry server-side.
              useAppStore.getState().incrementSavings(usage);
            }
            // Hold the final live numbers visible for a beat so the panel
            // doesn't flash to 0 between the SSE close and the next
            // /v1/telemetry/energy poll picking up the persisted record.
            window.setTimeout(() => {
              useAppStore.getState().setLiveEnergy(null);
            }, 1500);
            break;
          }
        }
      } else {
      for await (const sseEvent of streamChat(
        { model: selectedModel, messages: apiMessages, stream: true, temperature, max_tokens: maxTokens },
        controller.signal,
      )) {
        const eventName = sseEvent.event;

        if (eventName === 'agent_turn_start') {
          setStreamState({ phase: 'Agent thinking...' });
        } else if (eventName === 'inference_start') {
          setStreamState({ phase: 'Generating...' });
          useAppStore.getState().addLogEntry({
            timestamp: Date.now(), level: 'info', category: 'chat',
            message: `Generating with ${selectedModel}...`,
          });
        } else if (eventName === 'tool_call_start') {
          try {
            const data = JSON.parse(sseEvent.data);
            const tc: ToolCallInfo = {
              id: generateId(),
              tool: data.tool,
              arguments: serializeToolCallArguments(data.arguments),
              status: 'running',
            };
            toolCalls.push(tc);
            setStreamState({
              phase: `Using ${formatToolName(data.tool)}...`,
              activeToolCalls: [...toolCalls],
            });
            updateLastAssistant(convId, accumulatedContent, [...toolCalls]);
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(), level: 'info', category: 'tool',
              message: `Calling ${data.tool}(${serializeToolCallArguments(data.arguments)})`,
            });
          } catch {}
        } else if (eventName === 'tool_call_end') {
          try {
            const data = JSON.parse(sseEvent.data);
            const tc = toolCalls.find(
              (t) => t.tool === data.tool && t.status === 'running',
            );
            if (tc) {
              tc.status = data.success ? 'success' : 'error';
              tc.latency = data.latency;
              tc.result = data.result;
            }
            setStreamState({
              phase: 'Generating...',
              activeToolCalls: [...toolCalls],
            });
            updateLastAssistant(convId, accumulatedContent, [...toolCalls]);
          } catch {}
        } else {
          try {
            const data = JSON.parse(sseEvent.data);
            const delta = data.choices?.[0]?.delta;
            if (data.usage) usage = data.usage;
            if (data.complexity) complexity = data.complexity;
            routedEngine = engineFromCompletionChunk(data) ?? routedEngine;
            if (delta?.content) {
              if (!ttftMs) ttftMs = Date.now() - startTime;
              accumulatedContent += delta.content;

              if (ttsEnabledRef.current) {
                ttsBufferRef.current += delta.content;

                let sentenceMatch: RegExpMatchArray | null;

                while (
                  (sentenceMatch = ttsBufferRef.current.match(
                    /^([\s\S]*?[.!?])(?:\s+|$)/,
                  ))
                ) {
                  const sentence = sentenceMatch[1].trim();

                  if (sentence) {
                    enqueueTTS(sentence);
                  }

                  ttsBufferRef.current = ttsBufferRef.current.slice(
                    sentenceMatch[0].length,
                  );
                }
              }

              setStreamState({ content: accumulatedContent, phase: '' });

              const now = Date.now();
              if (now - lastFlush >= 80) {
                updateLastAssistant(
                  convId,
                  accumulatedContent,
                  toolCalls.length > 0 ? [...toolCalls] : undefined,
                );
                lastFlush = now;
              }
            }
            if (data.choices?.[0]?.finish_reason === 'stop') break;
          } catch {}
        }
      }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        generationAborted = true;
        // User cancelled or model switch — keep whatever was accumulated
        if (!accumulatedContent) accumulatedContent = '(Generation stopped)';
      } else {
        const errMsg = err?.message || String(err);
        accumulatedContent =
          accumulatedContent || `Error: ${errMsg}`;
        useAppStore.getState().addLogEntry({
          timestamp: Date.now(), level: 'error', category: 'chat',
          message: `Stream error: ${errMsg}`,
        });
      }
      // If we tore out mid-research, make sure the live System panel
      // numbers don't get stuck on the last sample.
      useAppStore.getState().setLiveEnergy(null);
    } finally {
      if (!accumulatedContent) {
        accumulatedContent = 'No response was generated. Please try again.';
      }

      if (
        ttsEnabledRef.current &&
        !generationAborted &&
        ttsBufferRef.current.trim()
      ) {
        enqueueTTS(ttsBufferRef.current);
        ttsBufferRef.current = '';
      }

      const totalMs = Date.now() - startTime;
      const appState = useAppStore.getState();
      const selectedOwner = appState.models.find((m) => m.id === selectedModel)?.owned_by;
      const engineLabel = resolveChatEngine({
        routedEngine,
        serverEngine: appState.serverInfo?.engine,
        selectedModel,
        selectedOwner,
      });
      const telemetry: MessageTelemetry = {
        engine: engineLabel,
        model_id: selectedModel,
        total_ms: totalMs,
        ttft_ms: ttftMs,
        tokens_per_sec: usage?.completion_tokens
          ? usage.completion_tokens / (totalMs / 1000)
          : undefined,
        complexity_score: complexity?.score,
        complexity_tier: complexity?.tier,
        suggested_max_tokens: complexity?.suggested_max_tokens,
      };

      updateLastAssistant(
        convId,
        accumulatedContent,
        toolCalls.length > 0 ? toolCalls : undefined,
        usage,
        telemetry,
        undefined,
        researchTraces.length > 0 ? researchTraces : undefined,
        researchSourcesByRef.size > 0 ? flushSources() : undefined,
      );
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      resetStream();
      useAppStore.getState().addLogEntry({
        timestamp: Date.now(), level: 'info', category: 'chat',
        message: `Response: ${accumulatedContent.length} chars`,
      });
      abortRef.current = null;

      // Research path updates session counters optimistically from the
      // `done` event's usage payload — re-fetching here would overwrite
      // it with a potentially stale snapshot if the server's research
      // telemetry hasn't been merged into /v1/savings yet.
      if (!deepResearch) {
        fetchSavings()
          .then((data) => useAppStore.getState().setSavings(data))
          .catch(() => {});
      }
    }
  }, [
    input,
    activeId,
    selectedModel,
    streamState.isStreaming,
    createConversation,
    addMessage,
    updateLastAssistant,
    setStreamState,
    resetStream,
    deepResearch,
    speechEnabled,
    ttsEnabled,
    temperature,
    maxTokens,
  ]);
      const visualizerProcessing =
      isCurrentChatStreaming && !ttsBusy;

    const visualizerDeepProcessing =
      visualizerProcessing && streamState.elapsedMs >= 3000;

    const visualizerIntensiveProcessing =
      visualizerProcessing && streamState.elapsedMs >= 7000;

    const visualizerStatus =
      ttsBusy
        ? 'J.A.R.V.I.S - SPEAKING'
        : visualizerIntensiveProcessing
          ? 'INTENSIVE PROCESSING'
          : visualizerDeepProcessing
            ? 'DEEP PROCESSING'
            : visualizerProcessing
              ? (streamState.phase || 'PROCESSING')
              : undefined;
  const handleMicClick = useCallback(async () => {
    if (speechState === 'recording') {
      try {
        const text = await stopRecording();

        if (text?.trim()) {
          await sendMessage(text);
        }
      } catch {
        // Error is captured in useSpeech
      }
    } else {
      await startRecording();
    }
  }, [speechState, startRecording, stopRecording, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="px-4 pb-4 pt-2" style={{ maxWidth: 'var(--chat-max-width)', margin: '0 auto', width: '100%' }}>
      <div className="mb-2 flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setDeepResearch(!deepResearch)}
            disabled={streamState.isStreaming}
            aria-pressed={deepResearch}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-colors cursor-pointer disabled:cursor-default disabled:opacity-50"
            style={{
              background: deepResearch ? 'var(--color-accent-subtle)' : 'transparent',
              border: `1px solid ${deepResearch ? 'var(--color-accent)' : 'var(--color-border)'}`,
              color: deepResearch ? 'var(--color-accent)' : 'var(--color-text-tertiary)',
            }}
            title={deepResearch ? 'Deep Research: on' : 'Deep Research: off'}
          >
            <JarvisVisualizer
              speaking={ttsBusy}
              processing={visualizerProcessing}
              deepProcessing={visualizerDeepProcessing}
              intensiveProcessing={visualizerIntensiveProcessing}
              status={visualizerStatus}
/>
            <Search size={12} />
            Deep Research
          </button>
        </div>
        {deepResearch && corpusSync.syncing && corpusSync.itemsSynced > 0 && (
          <div
            className="text-[11px] leading-snug"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            Searching over{' '}
            <span key={corpusSync.itemsSynced} className="sync-bump" style={{ color: 'var(--color-text-secondary)' }}>
              {corpusSync.itemsSynced.toLocaleString()}
            </span>{' '}
            items — sync in progress, results will improve as more data is indexed.
          </div>
        )}
      </div>
      <div
        className="jarvis-composer flex items-center gap-2 rounded-2xl px-4 py-3 transition-shadow"
        style={{
          background: 'var(--color-input-bg)',
          border: '1px solid var(--color-input-border)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={selectedModel ? 'Message OpenJarvis...' : 'Pick a model first (⌘K)...'}
          rows={1}
          className="flex-1 bg-transparent outline-none resize-none text-sm leading-relaxed"
          style={{ color: 'var(--color-text)', maxHeight: '200px' }}
          disabled={streamState.isStreaming || modelLoading}
        />
        {isCurrentChatStreaming ? (
          <button
            onClick={stopStreaming}
            className="p-2 rounded-xl transition-colors shrink-0 cursor-pointer"
            style={{ background: 'var(--color-error)', color: 'var(--color-on-accent)' }}
            title="Stop generating"
          >
            <Square size={16} />
          </button>
        ) : (
          <div className="flex items-center gap-1">
            <MicButton
              state={speechState}
              onClick={handleMicClick}
              disabled={micDisabled}
              reason={micReason}
            />
            <button
              onClick={() => sendMessage()}
              disabled={streamState.isStreaming || !input.trim() || modelLoading || !selectedModel}
              title={selectedModel ? 'Send message' : 'Pick a model first (⌘K)'}
              className="p-2 rounded-xl transition-colors shrink-0 cursor-pointer disabled:opacity-30 disabled:cursor-default"
              style={{
                background: input.trim() ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
                color: input.trim() ? 'white' : 'var(--color-text-tertiary)',
              }}
            >
              <Send size={16} />
            </button>
          </div>
        )}
      </div>
      <div className="flex items-center justify-center mt-2 text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
        <span>
          <kbd className="font-mono">Enter</kbd> to send &middot;{' '}
          <kbd className="font-mono">Shift+Enter</kbd> for new line
        </span>
      </div>
    </div>
  );
}