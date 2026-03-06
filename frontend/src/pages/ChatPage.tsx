/** Main chat page component with topic sidebar */

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { TraditionalChatPanel } from '../components/conversation';
import { FloatingActionButton } from '../components/ui';
import { DiaryListModal, DiaryDetailModal, DiaryEditModal } from '../components/diary';
import { TopicSidebar } from '../components/topics';
import { CharacterSelector } from '../components/character/CharacterSelector';
import { useChat } from '../hooks/useChat';
import { useTopics } from '../hooks/useTopics';
import { useCharacter } from '../hooks/useCharacter';
import backgroundImage from '/background/image.png';
import type { DiaryEntry } from '../services/diaryService';
import type { DisplayMessage } from '../types/chat';

const DEFAULT_CHARACTER_ID = 'sister_001';

export const ChatPage: React.FC = () => {
  const navigate = useNavigate();
  const [showDiaries, setShowDiaries] = useState(false);
  const [showDiaryDetail, setShowDiaryDetail] = useState(false);
  const [showDiaryEdit, setShowDiaryEdit] = useState(false);
  const [selectedDiary, setSelectedDiary] = useState<DiaryEntry | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Character selection state - character_id IS the UUID
  const [selectedCharacterId, setSelectedCharacterId] = useState(() => {
    const stored = localStorage.getItem('selectedCharacterId');
    if (stored && stored.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
      return stored;
    }
    return DEFAULT_CHARACTER_ID;
  });

  const handleCharacterChange = useCallback((characterId: string) => {
    setSelectedCharacterId(characterId);
    localStorage.setItem('selectedCharacterId', characterId);
  }, []);

  const handleVoiceInputStart = useCallback(() => {
    // Voice input start handling
  }, []);

  const handleVoiceInputEnd = useCallback(() => {
    // Voice input end handling
  }, []);

  // Topic management
  const {
    topics,
    currentTopicId,
    loading: topicsLoading,
    createNewTopic,
    selectTopic,
    deleteTopicById: deleteTopic,
    refreshTopics,
  } = useTopics({
    characterId: selectedCharacterId,
    onTopicChange: handleTopicChange,
  });

  // Chat functionality with topic support
  const {
    messages,
    loading,
    streamingMessage,
    sendStream,
    clearHistory,
    autoPlayTTS,
    toggleAutoPlayTTS,
    setMessages: setChatMessages,
  } = useChat({
    characterId: selectedCharacterId,
    topicId: currentTopicId ?? undefined,
  });

  // Get current character info
  const { character } = useCharacter(selectedCharacterId);

  // Handle topic selection
  async function handleTopicChange(topicId: number | null, topicMessages: DisplayMessage[]) {
    if (topicId === null) {
      // New topic, clear messages
      clearHistory();
    } else {
      // Load topic messages
      setChatMessages(topicMessages);
    }
  }

  // Create a new topic
  const handleCreateTopic = useCallback(async () => {
    const newTopicId = await createNewTopic();
    if (newTopicId !== null) {
      // New topic created, messages should be cleared
      clearHistory();
    }
  }, [createNewTopic, clearHistory]);

  // Handle topic selection from sidebar
  const handleSelectTopic = useCallback(async (topicId: number) => {
    await selectTopic(topicId);
  }, [selectTopic]);

  // Handle topic deletion
  const handleDeleteTopic = useCallback(async (topicId: number) => {
    await deleteTopic(topicId);
    // Refresh topics after deletion
    await refreshTopics();
  }, [deleteTopic, refreshTopics]);

  const handleSendMessage = (content: string) => {
    if (content.trim()) {
      sendStream(content.trim());
    }
  };

  const handleSelectDiary = (diary: DiaryEntry) => {
    setSelectedDiary(diary);
    setShowDiaries(false);
    setShowDiaryDetail(true);
  };

  const handleEditDiary = (diary: DiaryEntry) => {
    setSelectedDiary(diary);
    setShowDiaries(false);
    setShowDiaryDetail(false);
    setShowDiaryEdit(true);
  };

  const handleDiaryUpdate = (updatedDiary: DiaryEntry) => {
    setSelectedDiary(updatedDiary);
    // Refresh diary list if needed
  };

  return (
    <div className="h-screen flex relative">
      {/* Background image */}
      <div
        className="absolute inset-0 -z-10"
        style={{
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        }}
      />

      {/* Topic Sidebar */}
      <TopicSidebar
        topics={topics}
        currentTopicId={currentTopicId}
        loading={topicsLoading}
        collapsed={sidebarCollapsed}
        onSelectTopic={handleSelectTopic}
        onCreateTopic={handleCreateTopic}
        onDeleteTopic={handleDeleteTopic}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Top bar with character selector and management button */}
        <div className="absolute top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 bg-gradient-to-b from-white/80 to-transparent">
          <div className="flex items-center gap-3">
            <CharacterSelector
              selectedCharacterId={selectedCharacterId}
              onCharacterChange={handleCharacterChange}
            />
          </div>
          <button
            type="button"
            onClick={() => navigate('/characters')}
            className="px-4 py-2 rounded-full text-xs font-semibold border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
          >
            角色管理
          </button>
        </div>

        {/* Traditional Style Chat Panel - Full height */}
        <TraditionalChatPanel
          messages={messages}
          loading={loading}
          streamingMessage={streamingMessage}
          onSendMessage={handleSendMessage}
          onVoiceInputStart={handleVoiceInputStart}
          onVoiceInputEnd={handleVoiceInputEnd}
          placeholder="聊聊天吧～"
          characterId={selectedCharacterId}
        />

        {/* TTS Toggle */}
        <div className="fixed top-[72px] right-4 z-50">
          <button
            onClick={() => toggleAutoPlayTTS(!autoPlayTTS)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-semibold cursor-pointer transition-all duration-300 border-2 shadow-lg hover:-translate-y-0.5 hover:shadow-xl active:translate-y-0 bg-gradient-to-r from-pink-300 to-pink-500 border-pink-200 text-white shadow-pink-300/30 hover:from-pink-500 hover:to-pink-600 hover:shadow-pink-400/40"
            title={autoPlayTTS ? '关闭自动播放' : '开启自动播放'}
            type="button"
          >
            <span className="text-[13px] font-semibold hidden md:inline">
              {autoPlayTTS ? '语音开启' : '语音关闭'}
            </span>
          </button>
        </div>

        {/* Floating Buttons */}
        <FloatingActionButton
          onClick={() => setShowDiaries(true)}
          icon="📔"
          ariaLabel="日记本"
          position="bottom-right"
          index={0}
        />
        <DiaryListModal
          characterId={selectedCharacterId}
          characterName={character?.name}
          isOpen={showDiaries}
          onClose={() => setShowDiaries(false)}
          onSelectDiary={handleSelectDiary}
          onEditDiary={handleEditDiary}
        />
        <DiaryDetailModal
          diary={selectedDiary}
          isOpen={showDiaryDetail}
          onClose={() => setShowDiaryDetail(false)}
          characterName={character?.name}
        />
        <DiaryEditModal
          isOpen={showDiaryEdit}
          onClose={() => setShowDiaryEdit(false)}
          diary={selectedDiary}
          onUpdate={handleDiaryUpdate}
        />
      </div>
    </div>
  );
};
