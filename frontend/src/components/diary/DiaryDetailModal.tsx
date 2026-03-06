/** Diary detail modal component - Refined elegant style */

import React from 'react';
import { DiaryEntry, extractDateFromPath } from '../../services/diaryService';

interface DiaryDetailModalProps {
  diary: DiaryEntry | null;
  isOpen: boolean;
  onClose: () => void;
  characterName?: string;
}

export const DiaryDetailModal: React.FC<DiaryDetailModalProps> = ({
  diary,
  isOpen,
  onClose,
  characterName
}) => {
  if (!isOpen || !diary) return null;

  // Extract date from path
  const date = extractDateFromPath(diary.path);

  return (
    <div
      className="fixed inset-0 bg-black/20 backdrop-blur-sm animate-fade-in z-50"
      onClick={onClose}
    >
      <div
        className="mx-4 my-8 max-w-2xl bg-white dark:bg-neutral-800 rounded-3xl shadow-xl border border-neutral-200 dark:border-neutral-700 max-h-[calc(100vh-4rem)] flex flex-col animate-message-in overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-rose-50 to-rose-100/50 dark:from-rose-950/30 dark:to-transparent px-6 py-5 border-b border-rose-100 dark:border-neutral-700">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl">📔</span>
                <h2 className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">{characterName ? `${characterName}的日记` : '日记'}</h2>
              </div>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                {date.toLocaleDateString('zh-CN', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                  weekday: 'long'
                })}
              </p>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded-full text-neutral-400 hover:text-neutral-600 hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
              aria-label="关闭"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 scrollbar-elegant">
          {/* Diary content */}
          <div className="mb-6">
            <div className="prose prose-sm max-w-none">
              <p className="text-neutral-700 dark:text-neutral-300 leading-loose whitespace-pre-wrap text-base">
                {diary.content}
              </p>
            </div>
          </div>

          {/* Metadata */}
          <div className="pt-4 border-t border-neutral-200 dark:border-neutral-700 text-xs text-neutral-500 dark:text-neutral-400 space-y-1">
            <p>角色 ID: {diary.character_id}</p>
            <p>文件路径: {diary.path}</p>
            <p>修改时间: {new Date(diary.mtime * 1000).toLocaleString('zh-CN')}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
