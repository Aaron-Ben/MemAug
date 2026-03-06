/** Traditional Chat Panel Component - Refined Elegant Style */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { AudioRecorder, recognizeFromBlob } from '../../services/voiceService';
import type { DisplayMessage, RecordingState } from '../../types/chat';
import clsx from 'clsx';
import { AIMessageBubble, AILoadingBubble, UserMessageBubble } from '../chat';

interface TraditionalChatPanelProps {
  messages: DisplayMessage[];
  loading: boolean;
  streamingMessage: string;
  onSendMessage: (content: string) => void;
  onVoiceInputStart?: () => void;
  onVoiceInputEnd?: () => void;
  placeholder?: string;
  characterId?: string;
}

export const TraditionalChatPanel: React.FC<TraditionalChatPanelProps> = ({
  messages,
  loading,
  streamingMessage,
  onSendMessage,
  onVoiceInputStart,
  onVoiceInputEnd,
  placeholder = '聊聊天吧～',
  characterId,
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [recordingTime, setRecordingTime] = useState(0);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 120);
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [input]);

  // Recording timer
  useEffect(() => {
    let interval: number | null = null;
    if (recordingState === 'recording') {
      interval = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      setRecordingTime(0);
    }
    return () => {
      if (interval !== null) {
        clearInterval(interval);
      }
    };
  }, [recordingState]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = useCallback(() => {
    const trimmedInput = input.trim();
    if (trimmedInput && !loading && recordingState === 'idle') {
      onSendMessage(trimmedInput);
      setInput('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [input, loading, recordingState, onSendMessage]);

  const handleVoiceStart = useCallback(async () => {
    if (loading || recordingState !== 'idle') return;

    try {
      const recorder = new AudioRecorder();
      recorderRef.current = recorder;
      await recorder.start();

      setRecordingState('recording');
      onVoiceInputStart?.();
    } catch (error) {
      console.error('Failed to start recording:', error);
      recorderRef.current = null;
    }
  }, [loading, recordingState, onVoiceInputStart]);

  const handleVoiceEnd = useCallback(async () => {
    if (recordingState !== 'recording' || !recorderRef.current) return;

    setRecordingState('processing');

    try {
      const audioBlob = await recorderRef.current.stop();
      recorderRef.current = null;

      const result = await recognizeFromBlob(audioBlob, { characterId: characterId || 'sister_001' });

      if (result.success && result.text) {
        const newText = input + (input ? ' ' : '') + result.text;
        setInput(newText);
      } else if (result.error) {
        console.error('Voice recognition failed:', result.error);
      }
    } catch (error) {
      console.error('Voice input error:', error);
    } finally {
      setRecordingState('idle');
      onVoiceInputEnd?.();
    }
  }, [recordingState, input, onVoiceInputEnd, characterId]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col h-full bg-transparent">
      {/* Messages Area */}
      <div
        className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth scrollbar-elegant"
        ref={messagesContainerRef}
      >
        <div className="max-w-2xl mx-auto flex flex-col gap-4">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center min-h-[300px] text-neutral-400 dark:text-neutral-500 gap-4">
              <div className="text-6xl opacity-60">🌸</div>
              <div className="text-base">开始聊天吧～</div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={clsx(
                'flex animate-message-in',
                message.isUser ? 'justify-end' : 'justify-start'
              )}
            >
              {message.isUser ? (
                <UserMessageBubble content={message.content} timestamp={message.timestamp} />
              ) : (
                <AIMessageBubble content={message.content} timestamp={message.timestamp} />
              )}
            </div>
          ))}

          {/* Streaming message */}
          {streamingMessage && (
            <div className="flex justify-start animate-message-in">
              <AIMessageBubble content={streamingMessage} isStreaming={true} />
            </div>
          )}

          {/* Loading indicator */}
          {loading && !streamingMessage && (
            <div className="flex justify-start animate-message-in">
              <AILoadingBubble />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 md:p-6 bg-gradient-to-t from-rose-50/80 to-transparent dark:from-dark-secondary/50 backdrop-blur-sm border-t border-rose-100/50 dark:border-neutral-800">
        <div className="mx-auto max-w-2xl flex gap-3 items-end bg-white dark:bg-neutral-800 rounded-full px-5 py-3 shadow-md border border-neutral-200 dark:border-neutral-700 focus-within:border-rose-400 dark:focus-within:border-rose-500 focus-within:shadow-lg transition-all duration-200">
          <textarea
            ref={textareaRef}
            className="flex-1 border-none outline-none text-base bg-transparent text-neutral-800 dark:text-neutral-100 resize-none min-h-[24px] max-h-[120px] leading-relaxed placeholder:text-neutral-400 disabled:opacity-60 disabled:cursor-not-allowed"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={loading || recordingState === 'recording'}
            rows={1}
          />
          <div className="flex items-center gap-2">
            {/* Voice input button */}
            <button
              className={clsx(
                'w-10 h-10 rounded-full bg-emerald-400 hover:bg-emerald-500 dark:bg-emerald-500 dark:hover:bg-emerald-600 text-white flex items-center justify-center shrink-0 transition-all duration-200 text-lg disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]',
                recordingState === 'recording' && 'bg-rose-400 hover:bg-rose-500 dark:bg-rose-500 dark:hover:bg-rose-600 w-auto px-3 rounded-2xl'
              )}
              onMouseDown={handleVoiceStart}
              onMouseUp={handleVoiceEnd}
              onMouseLeave={handleVoiceEnd}
              onTouchStart={(e) => { e.preventDefault(); handleVoiceStart(); }}
              onTouchEnd={(e) => { e.preventDefault(); handleVoiceEnd(); }}
              disabled={loading || recordingState !== 'idle'}
              title={
                recordingState === 'recording' ? '松开结束录音' :
                recordingState === 'processing' ? '识别中...' :
                '按住说话'
              }
              type="button"
            >
              {recordingState === 'recording' ? (
                <>
                  <span>🎤</span>
                  <span className="text-[13px] font-semibold tabular-nums ml-1">{formatTime(recordingTime)}</span>
                </>
              ) : recordingState === 'processing' ? (
                <span className="animate-spin text-lg">⌛</span>
              ) : (
                <span>🎤</span>
              )}
            </button>

            {/* Send button */}
            <button
              className="w-10 h-10 rounded-full bg-gradient-to-r from-rose-400 to-rose-500 hover:from-rose-500 hover:to-rose-600 text-white flex items-center justify-center gap-1 shrink-0 text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98] shadow-sm hover:shadow-md"
              onClick={handleSend}
              disabled={!input.trim() || loading || recordingState !== 'idle'}
              title="发送"
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
