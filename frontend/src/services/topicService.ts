/** Topic API service for chat history management */

import { API_ENDPOINTS, apiRequest } from './api';
import type {
  Topic,
  TopicResponse,
  TopicListResponse,
  ChatHistoryResponse,
} from '../types/chat';

export interface CreateTopicRequest {
  character_id?: string;
}

/**
 * Create a new topic
 */
export async function createTopic(request: CreateTopicRequest): Promise<Topic> {
  const response = await apiRequest<TopicResponse>(API_ENDPOINTS.topicCreate(), {
    method: 'POST',
    body: JSON.stringify(request),
  });
  return {
    topic_id: response.topic_id,
    character_id: response.character_id,
    created_at: new Date(response.created_at),
    updated_at: new Date(response.updated_at),
    message_count: response.message_count,
  };
}

/**
 * List all topics, optionally filtered by character ID
 */
export async function listTopics(characterId?: string): Promise<TopicListResponse> {
  const url = API_ENDPOINTS.topicList(characterId);
  return apiRequest<TopicListResponse>(url);
}

/**
 * Delete a topic
 */
export async function deleteTopic(
  topicId: number,
  characterId?: string
): Promise<void> {
  const url = API_ENDPOINTS.topicDelete(topicId);
  const body = characterId ? { character_id: characterId } : undefined;
  await apiRequest<void>(url, {
    method: 'DELETE',
    ...(body && { body: JSON.stringify(body) }),
  });
}

/**
 * Get message history for a specific topic
 */
export async function getTopicHistory(
  topicId: number,
  characterId?: string
): Promise<ChatHistoryResponse> {
  const url = API_ENDPOINTS.topicHistory(topicId, characterId);
  return apiRequest<ChatHistoryResponse>(url);
}

/**
 * Format a timestamp to relative time string
 */
export function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - new Date(date).getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return '刚刚';
  if (diffMins < 60) return `${diffMins}分钟前`;
  if (diffHours < 24) return `${diffHours}小时前`;
  if (diffDays < 7) return `${diffDays}天前`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}周前`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)}月前`;
  return `${Math.floor(diffDays / 365)}年前`;
}
