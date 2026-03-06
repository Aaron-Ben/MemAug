/** Custom hook for chat functionality */

import { useState, useCallback, useRef } from 'react';
import { sendMessage, sendMessageStream, getChatStarter } from '../services/chatService';
import { speakText } from '../services/ttsService';
import type { DisplayMessage, ChatRequest, Message } from '../types/chat';

const DEFAULT_CHARACTER_ID = 'sister_001';
const STORAGE_KEY = 'chat_history';

interface UseChatOptions {
  characterId?: string;
  topicId?: number;
}

export function useChat(options?: UseChatOptions) {
  const {
    characterId = DEFAULT_CHARACTER_ID,
    topicId,
  } = options || {};

  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const errorRef = useRef<string | null>(null);

  // Load history from localStorage on mount (only if no topic is set)
  useState(() => {
    // Only load from localStorage if we're not using topics
    if (!topicId) {
      try {
        const stored = localStorage.getItem(`${STORAGE_KEY}_${characterId}`);
        if (stored) {
          const parsed = JSON.parse(stored);
          setMessages(parsed.map((msg: DisplayMessage) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })));
        }
      } catch (err) {
        console.error('Failed to load chat history:', err);
      }
    }
  });

  // Save to localStorage whenever messages change
  const saveHistory = useCallback((newMessages: DisplayMessage[]) => {
    try {
      localStorage.setItem(
        `${STORAGE_KEY}_${characterId}`,
        JSON.stringify(newMessages)
      );
    } catch (err) {
      console.error('Failed to save chat history:', err);
    }
  }, [characterId]);

  const addMessage = useCallback((message: DisplayMessage) => {
    setMessages((prev) => {
      const newMessages = [...prev, message];
      saveHistory(newMessages);
      return newMessages;
    });
  }, [saveHistory]);

  const send = useCallback(async (content: string) => {
    if (!content.trim() || loading) return;

    setLoading(true);
    errorRef.current = null;

    // Add user message
    const userMessage: DisplayMessage = {
      id: `user-${Date.now()}`,
      content: content.trim(),
      isUser: true,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    // Build conversation history for API
    const conversationHistory: Message[] = messages
      .slice(-10) // Last 10 messages for context
      .map((msg) => ({
        role: msg.isUser ? 'user' : 'assistant',
        content: msg.content,
      }));

    const request: ChatRequest = {
      message: content,
      character_id: characterId,
      conversation_history: conversationHistory,
    };

    try {
      const response = await sendMessage(request, {
        topic_id: topicId,
      });

      const assistantMessage: DisplayMessage = {
        id: `assistant-${Date.now()}`,
        content: response.message,
        isUser: false,
        timestamp: new Date(response.timestamp),
        emotion: response.emotion_detected,
      };
      addMessage(assistantMessage);
    } catch (err) {
      errorRef.current = err instanceof Error ? err.message : 'Failed to send message';
    } finally {
      setLoading(false);
    }
  }, [messages, characterId, topicId, loading, addMessage]);

  const sendStream = useCallback(async (content: string) => {
    if (!content.trim() || loading) return;

    setLoading(true);
    errorRef.current = null;
    setStreamingMessage('');

    // Add user message
    const userMessage: DisplayMessage = {
      id: `user-${Date.now()}`,
      content: content.trim(),
      isUser: true,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    // Build conversation history
    const conversationHistory: Message[] = messages
      .slice(-10)
      .map((msg) => ({
        role: msg.isUser ? 'user' : 'assistant',
        content: msg.content,
      }));

    const request: ChatRequest = {
      message: content,
      character_id: characterId,
      conversation_history: conversationHistory,
    };

    // Create placeholder for streaming message
    const streamingId = `streaming-${Date.now()}`;
    setMessages((prev) => [...prev, {
      id: streamingId,
      content: '',
      isUser: false,
      timestamp: new Date(),
    }]);

    try {
      let fullResponse = '';
      for await (const chunk of sendMessageStream(request, {
        topic_id: topicId,
      })) {
        fullResponse += chunk;
        setStreamingMessage(fullResponse);
        setMessages((prev) => prev.map((msg) =>
          msg.id === streamingId
            ? { ...msg, content: fullResponse }
            : msg
        ));
      }

      // Finalize the message
      setMessages((prev) => prev.map((msg) =>
        msg.id === streamingId
          ? { ...msg, id: `assistant-${Date.now()}` }
          : msg
      ));

      saveHistory(messages);
    } catch (err) {
      errorRef.current = err instanceof Error ? err.message : 'Failed to send message';
      // Remove the streaming message placeholder
      setMessages((prev) => prev.filter((msg) => msg.id !== streamingId));
    } finally {
      setLoading(false);
      setStreamingMessage('');
    }
  }, [messages, characterId, topicId, loading, addMessage, saveHistory]);

  const clearHistory = useCallback(() => {
    setMessages([]);
    // Only clear localStorage if we're not using topics
    if (!topicId) {
      localStorage.removeItem(`${STORAGE_KEY}_${characterId}`);
    }
  }, [characterId, topicId]);

  const getStarter = useCallback(async () => {
    try {
      const data = await getChatStarter(characterId);
      return data.starter;
    } catch (err) {
      console.error('Failed to get conversation starter:', err);
      return null;
    }
  }, [characterId]);

  // TTS: Play TTS for given text
  const playTTS = useCallback(async (text: string) => {
    try {
      await speakText(text, 'genie', characterId);
    } catch (err) {
      console.error('TTS playback failed:', err);
    }
  }, [characterId]);

  // Set messages directly (used when loading topic history)
  const setMessagesDirect = useCallback((newMessages: DisplayMessage[]) => {
    setMessages(newMessages);
  }, []);

  // Get current topic ID
  const getCurrentTopicId = useCallback(() => topicId, [topicId]);

  // TTS auto-play state
  const [autoPlayTTS, setAutoPlayTTS] = useState(() => {
    // Load from localStorage
    try {
      const stored = localStorage.getItem('autoplay_tts');
      return stored === 'true';
    } catch {
      return false;
    }
  });

  // Toggle auto-play TTS
  const toggleAutoPlayTTS = useCallback((enabled: boolean) => {
    setAutoPlayTTS(enabled);
    try {
      localStorage.setItem('autoplay_tts', String(enabled));
    } catch (err) {
      console.error('Failed to save TTS preference:', err);
    }
  }, []);

  return {
    messages,
    loading,
    error: errorRef.current,
    streamingMessage,
    send,
    sendStream,
    clearHistory,
    getStarter,
    // TTS
    autoPlayTTS,
    toggleAutoPlayTTS,
    playTTS,
    // Topic support
    setMessages: setMessagesDirect,
    getCurrentTopicId,
  };
}
