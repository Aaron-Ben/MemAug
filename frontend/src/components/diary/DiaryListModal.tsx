/** Diary list modal component - Refined elegant style */

import React, { useEffect, useState, useCallback } from 'react';
import { listDiaries, deleteDiary, type DiaryEntry, extractDateFromPath } from '../../services/diaryService';
import { DiaryDeleteModal } from './DiaryDeleteModal';
import { DiaryTimeline } from './DiaryTimeline';

interface DiaryListModalProps {
  isOpen: boolean;
  onClose: () => void;
  characterId: string;
  characterName?: string;
  onSelectDiary?: (diary: DiaryEntry) => void;
  onEditDiary?: (diary: DiaryEntry) => void;
}

export const DiaryListModal: React.FC<DiaryListModalProps> = ({
  isOpen,
  onClose,
  characterId,
  characterName,
  onSelectDiary,
  onEditDiary
}) => {
  const [diaries, setDiaries] = useState<DiaryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [diaryToDelete, setDiaryToDelete] = useState<DiaryEntry | null>(null);

  const loadDiaries = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await listDiaries(characterId);
      setDiaries(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load diaries');
    } finally {
      setLoading(false);
    }
  }, [characterId]);

  useEffect(() => {
    if (isOpen) {
      loadDiaries();
    }
  }, [isOpen, loadDiaries]);

  const handleDeleteClick = (diary: DiaryEntry, e: React.MouseEvent) => {
    e.stopPropagation();
    setDiaryToDelete(diary);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async () => {
    if (!diaryToDelete) return;

    try {
      await deleteDiary(diaryToDelete.path);
      // Reload the list after deletion
      loadDiaries();
      setDiaryToDelete(null);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete diary');
    }
  };

  const handleEdit = (diary: DiaryEntry, e: React.MouseEvent) => {
    e.stopPropagation();
    onEditDiary?.(diary);
  };

  if (!isOpen) return null;

  return (
    <>
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm animate-fade-in z-50"
        onClick={onClose}
      >
        <div
          className="mx-4 my-8 bg-amber-50/95 dark:bg-stone-800/95 rounded-3xl shadow-xl border border-amber-100 dark:border-stone-600 max-w-5xl max-h-[calc(100vh-4rem)] flex flex-col animate-message-in overflow-hidden backdrop-blur-sm"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-amber-100/80 via-rose-50/60 to-amber-50/50 dark:from-stone-700/50 dark:via-rose-950/20 dark:to-stone-800/50 px-6 py-5 border-b border-amber-200/50 dark:border-stone-600">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-amber-900 dark:text-amber-100 flex items-center gap-2">
                  <span>📔</span>
                  {characterName || '日记本'}
                </h2>
                <p className="text-sm text-amber-700/80 dark:text-amber-300/70 mt-1">记录对话的点点滴滴～</p>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 flex items-center justify-center rounded-full text-amber-500/60 hover:text-amber-700 hover:bg-amber-100/60 dark:text-amber-400/60 dark:hover:text-amber-300 dark:hover:bg-amber-950/30 transition-colors"
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
            {loading ? (
              <div className="text-center text-amber-600/70 dark:text-amber-400/60 py-12">
                <div className="w-10 h-10 border-3 border-rose-200 border-t-rose-500 rounded-full animate-spin mx-auto mb-4"></div>
                <p>加载中...</p>
              </div>
            ) : error ? (
              <div className="text-center text-rose-600 dark:text-rose-400 py-12">
                <p>{error}</p>
              </div>
            ) : diaries.length === 0 ? (
              <div className="text-center text-amber-600/70 dark:text-amber-400/60 py-12">
                <p className="text-lg mb-2">还没有日记呢～</p>
                <p className="text-sm">会记录对话中的重要时刻</p>
              </div>
            ) : (
              <DiaryTimeline
                diaries={diaries}
                onSelectDiary={onSelectDiary || (() => {})}
                onEditDiary={handleEdit}
                onDeleteDiary={(diary, e) => handleDeleteClick(diary, e)}
              />
            )}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <DiaryDeleteModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleConfirmDelete}
        diaryDate={diaryToDelete ? extractDateFromPath(diaryToDelete.path).toLocaleDateString('zh-CN') : ''}
      />
    </>
  );
};
