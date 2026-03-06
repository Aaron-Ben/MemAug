/** Custom hook for topic management */

import { useState, useCallback, useEffect } from 'react';
import {
  createTopic,
  listTopics,
  deleteTopic,
  getTopicHistory,
} from '../services/topicService';
import type {
  TopicListItem,
  ChatHistoryResponse,
  ChatMessageResponse,
  DisplayMessage,
} from '../types/chat';

interface UseTopicsOptions {
  characterId: string;
  onTopicChange?: (topicId: number | null, messages: DisplayMessage[]) => void;
}

interface UseTopicsReturn {
  topics: TopicListItem[];
  currentTopicId: number | null;
  loading: boolean;
  error: string | null;
  createNewTopic: () => Promise<number | null>;
  selectTopic: (topicId: number) => Promise<void>;
  deleteTopicById: (topicId: number) => Promise<void>;
  refreshTopics: () => Promise<TopicListItem[]>;
  loadTopicMessages: (topicId: number) => Promise<DisplayMessage[]>;
}

const TOPIC_STORAGE_KEY = 'current_topic_id';

export function useTopics({ characterId, onTopicChange }: UseTopicsOptions): UseTopicsReturn {
  const [topics, setTopics] = useState<TopicListItem[]>([]);
  const [currentTopicId, setCurrentTopicId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load current topic from localStorage
  useEffect(() => {
    const storedTopicId = localStorage.getItem(TOPIC_STORAGE_KEY);
    if (storedTopicId) {
      setCurrentTopicId(parseInt(storedTopicId, 10));
    }
  }, []);

  // Load topics list
  const refreshTopics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listTopics(characterId);
      // Transform API response to TopicListItem format
      const { formatTimeAgo } = await import('../services/topicService');
      const topicListItems: TopicListItem[] = response.topics.map((topicResponse) => ({
        topic: {
          topic_id: topicResponse.topic_id,
          character_id: topicResponse.character_id,
          created_at: new Date(topicResponse.created_at),
          updated_at: new Date(topicResponse.updated_at),
          message_count: topicResponse.message_count,
        },
        preview: `${topicResponse.message_count > 0 ? '对话' : '新对话'} - ${topicResponse.message_count} 条消息`,
        timeAgo: formatTimeAgo(new Date(topicResponse.updated_at)),
      }));
      setTopics(topicListItems);
      return topicListItems;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load topics';
      setError(errorMessage);
      console.error('Failed to load topics:', err);
      return [];
    } finally {
      setLoading(false);
    }
  }, [characterId]);

  // Load topics when characterId changes
  useEffect(() => {
    const loadAndSelectFirstTopic = async () => {
      const topicList = await refreshTopics();
      // Auto-select the first topic for the new character
      // (if there's at least one topic and current topic doesn't belong to this character)
      if (topicList.length > 0) {
        const firstTopicId = topicList[0].topic.topic_id;
        // Check if current topic belongs to this character
        const currentTopicBelongsToThisChar = topicList.some(
          t => t.topic.topic_id === currentTopicId
        );
        if (!currentTopicBelongsToThisChar) {
          // Current topic doesn't belong to this character, switch to first topic
          setCurrentTopicId(firstTopicId);
          localStorage.setItem(TOPIC_STORAGE_KEY, String(firstTopicId));
          onTopicChange?.(firstTopicId, []);
        }
      } else {
        // No topics for this character, clear current topic
        setCurrentTopicId(null);
        localStorage.removeItem(TOPIC_STORAGE_KEY);
        onTopicChange?.(null, []);
      }
    };
    loadAndSelectFirstTopic();
  }, [characterId, refreshTopics]);

  // Convert API messages to DisplayMessage format
  const convertToDisplayMessages = useCallback((messages: ChatMessageResponse[]): DisplayMessage[] => {
    return messages.map((msg) => ({
      id: msg.id,
      content: msg.content,
      isUser: msg.role === 'user',
      timestamp: new Date(msg.timestamp),  // msg.timestamp is now in milliseconds
      name: msg.name,
    }));
  }, []);

  // Load messages for a specific topic
  const loadTopicMessages = useCallback(async (topicId: number): Promise<DisplayMessage[]> => {
    setLoading(true);
    setError(null);
    try {
      const response: ChatHistoryResponse = await getTopicHistory(topicId, characterId);
      const displayMessages = convertToDisplayMessages(response.messages);
      return displayMessages;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load topic history';
      setError(errorMessage);
      console.error('Failed to load topic history:', err);
      return [];
    } finally {
      setLoading(false);
    }
  }, [characterId, convertToDisplayMessages]);

  // Create a new topic
  const createNewTopic = useCallback(async (): Promise<number | null> => {
    setLoading(true);
    setError(null);
    try {
      const newTopic = await createTopic({
        character_id: characterId,
      });

      // Refresh the topics list
      await refreshTopics();

      // Set as current topic
      setCurrentTopicId(newTopic.topic_id);
      localStorage.setItem(TOPIC_STORAGE_KEY, String(newTopic.topic_id));

      // Notify parent of topic change with empty messages
      onTopicChange?.(newTopic.topic_id, []);

      return newTopic.topic_id;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create topic';
      setError(errorMessage);
      console.error('Failed to create topic:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [characterId, refreshTopics, onTopicChange]);

  // Select an existing topic
  const selectTopic = useCallback(async (topicId: number) => {
    const messages = await loadTopicMessages(topicId);
    setCurrentTopicId(topicId);
    localStorage.setItem(TOPIC_STORAGE_KEY, String(topicId));
    onTopicChange?.(topicId, messages);
  }, [loadTopicMessages, onTopicChange]);

  // Delete a topic
  const deleteTopicById = useCallback(async (topicId: number) => {
    setLoading(true);
    setError(null);
    try {
      await deleteTopic(topicId, characterId);

      // If we deleted the current topic, switch to another
      if (topicId === currentTopicId) {
        const remainingTopics = topics.filter((t) => t.topic.topic_id !== topicId);
        if (remainingTopics.length > 0) {
          const nextTopic = remainingTopics[0];
          await selectTopic(nextTopic.topic.topic_id);
        } else {
          // No topics left, create a new one
          await createNewTopic();
        }
      }

      // Refresh the list
      await refreshTopics();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete topic';
      setError(errorMessage);
      console.error('Failed to delete topic:', err);
    } finally {
      setLoading(false);
    }
  }, [characterId, currentTopicId, topics, selectTopic, createNewTopic, refreshTopics]);

  return {
    topics,
    currentTopicId,
    loading,
    error,
    createNewTopic,
    selectTopic,
    deleteTopicById,
    refreshTopics,
    loadTopicMessages,
  };
}
