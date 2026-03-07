/** Main chat page component with topic sidebar */

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { TraditionalChatPanel } from '../components/conversation';
import { DiaryListModal, DiaryDetailModal, DiaryEditModal } from '../components/diary';
import { TopicSidebar } from '../components/topics';
import { CharacterSelector } from '../components/character/CharacterSelector';
import { ThemeToggleButton } from '../components/ui/ThemeToggleButton';
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
    <div className="h-screen flex relative bg-gradient-to-br from-rose-50/30 via-white to-pink-50/20 dark:from-night-primary dark:via-night-secondary dark:to-night-primary">
      {/* Background Image - Filter 方案 */}
      <div className="absolute inset-0 -z-10">
        {/* 背景图片层 - 使用 filter 方案处理暗色模式 */}
        <div
          className="absolute inset-0 transition-all duration-700 ease-in-out"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
          }}
        >
          {/* 亮色模式 */}
          <div className="absolute inset-0 opacity-40" />
        </div>

        {/* 暗色模式滤镜层 */}
        <div
          className="absolute inset-0 opacity-0 dark:opacity-100 transition-opacity duration-700 ease-in-out pointer-events-none"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
            filter: 'brightness(0.7) grayscale(40%) contrast(0.9)',
          }}
        />

        {/* 统一的透明度控制层 */}
        <div className="absolute inset-0 opacity-40 dark:opacity-35 transition-opacity duration-700 ease-in-out bg-current pointer-events-none" />

        {/* 微妙的纹理层 */}
        <div
          className="absolute inset-0 opacity-[0.03] dark:opacity-[0.05] pointer-events-none transition-opacity duration-700"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
          }}
        />
      </div>

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
        {/* 顶部左右两侧按钮组 - 分散布局避免遮挡 */}
        {/* 左上角：角色选择器 */}
        <div className="absolute top-4 left-4 z-40">
          <CharacterSelector
            selectedCharacterId={selectedCharacterId}
            onCharacterChange={handleCharacterChange}
          />
        </div>

        {/* 右上角：设置和主题按钮组 */}
        <div className="absolute top-4 right-4 z-40 flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate('/characters')}
            className="flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-rose-200 dark:border-neutral-600 text-rose-700 dark:text-rose-200 hover:border-rose-300 dark:hover:border-rose-700 hover:shadow-md active:scale-95"
            title="角色管理"
          >
            <span>⚙️</span>
            <span className="hidden sm:inline">管理</span>
          </button>
          <ThemeToggleButton variant="minimal" size="sm" />
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
          characterName={character?.name}
        />

        {/* 右下角快捷操作区 - 垂直排列 */}
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 items-end">
          {/* TTS 语音开关 */}
          <button
            onClick={() => toggleAutoPlayTTS(!autoPlayTTS)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-semibold cursor-pointer transition-all duration-300 border-2 shadow-lg hover:-translate-y-0.5 hover:shadow-xl active:translate-y-0 active:scale-95 group",
              autoPlayTTS
                ? "bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 border-rose-300 text-white shadow-rose-soft hover:shadow-rose-soft-lg"
                : "bg-white/90 dark:bg-neutral-800/90 backdrop-blur-sm border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700"
            )}
            title={autoPlayTTS ? '关闭自动播放' : '开启自动播放'}
            type="button"
          >
            <span className={clsx("text-base transition-transform duration-300", autoPlayTTS && "group-hover:scale-110")}>🔊</span>
            <span className="text-[13px] font-semibold">
              {autoPlayTTS ? '语音开启' : '语音'}
            </span>
          </button>

          {/* 日记按钮 */}
          <button
            onClick={() => setShowDiaries(true)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-semibold cursor-pointer transition-all duration-300 border-2 shadow-lg hover:-translate-y-0.5 hover:shadow-xl active:translate-y-0 active:scale-95 group bg-gradient-to-br from-rose-400 via-rose-500 to-pink-500 border-rose-300 text-white shadow-rose-soft hover:shadow-rose-soft-lg"
            title="日记本"
            type="button"
          >
            <span className="text-base transition-transform duration-300 group-hover:rotate-12">📔</span>
            <span className="text-[13px] font-semibold">日记</span>
          </button>
        </div>
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
