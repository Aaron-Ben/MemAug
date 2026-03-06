/** API base configuration - Simplified for file system based storage */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  // User character management endpoints
  createCharacter: () => `${API_BASE_URL}/api/v1/character/create`,
  listUserCharacters: () => `${API_BASE_URL}/api/v1/character/user/list`,
  getUserCharacter: (id: string) => `${API_BASE_URL}/api/v1/character/user/${id}`,
  deleteUserCharacter: (id: string) => `${API_BASE_URL}/api/v1/character/user/${id}`,
  updateUserCharacterPrompt: (id: string) => `${API_BASE_URL}/api/v1/character/user/${id}`,

  // Chat endpoints
  chat: () => `${API_BASE_URL}/api/v1/chat/`,
  chatStream: () => `${API_BASE_URL}/api/v1/chat/stream`,
  chatStarter: () => `${API_BASE_URL}/api/v1/chat/starter`,
  voiceInput: () => `${API_BASE_URL}/api/v1/chat/voice`,
  voiceChat: () => `${API_BASE_URL}/api/v1/chat/voice/chat`,

  // TTS endpoints
  tts: () => `${API_BASE_URL}/api/v1/chat/tts`,
  ttsAudio: (filename: string) => `${API_BASE_URL}/api/v1/chat/tts/audio/${filename}`,

  // Topic endpoints
  topicList: (characterId?: string) =>
    `${API_BASE_URL}/api/v1/chat/topics${characterId ? `?character_id=${characterId}` : ''}`,
  topicCreate: () => `${API_BASE_URL}/api/v1/chat/topics`,
  topicDelete: (id: number) => `${API_BASE_URL}/api/v1/chat/topics/${id}`,
  topicHistory: (id: number, characterId?: string) =>
    `${API_BASE_URL}/api/v1/chat/topics/${id}/history${characterId ? `?character_id=${characterId}` : ''}`,
} as const;

export async function apiRequest<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  return response.json();
}

export default API_BASE_URL;
