/** Diary card component for displaying a single diary entry - Refined elegant style */

import React from 'react';
import { DiaryEntry, extractDateFromPath } from '../../services/diaryService';

interface DiaryCardProps {
  diary: DiaryEntry;
  onSelect: (diary: DiaryEntry) => void;
  onEdit: (diary: DiaryEntry, e: React.MouseEvent) => void;
  onDelete: (diary: DiaryEntry, e: React.MouseEvent) => void;
}

export const DiaryCard: React.FC<DiaryCardProps> = ({
  diary,
  onSelect,
  onEdit,
  onDelete,
}) => {
  // Extract date from path for display
  const date = extractDateFromPath(diary.path);
  const dateStr = date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  // Extract tag from content
  const tagMatch = diary.content.match(/Tag:\s*(.+)$/m);
  const tag = tagMatch ? tagMatch[1].trim().split(',').map(t => t.trim()).filter(Boolean)[0] : null;
  const displayContent = tagMatch
    ? diary.content.replace(/Tag:\s*(.+)$/m, '').trim()
    : diary.content;

  return (
    <div
      className="relative pl-6 pb-8 border-l border-amber-200/50 dark:border-stone-600 last:border-l-0 animate-message-in group cursor-pointer"
      onClick={() => onSelect(diary)}
    >
      {/* Date marker */}
      <div className="absolute -left-[5px] top-0 w-2 h-2 bg-amber-500 dark:bg-amber-600 rounded-full border-2 border-amber-50 dark:border-stone-800"></div>

      {/* Card */}
      <div className="bg-amber-50/90 dark:bg-stone-800/90 rounded-2xl p-5 shadow-sm border border-amber-200/50 dark:border-stone-600 hover:border-amber-300/70 dark:hover:border-amber-700/50 hover:shadow-md transition-all duration-200">
        {/* Action buttons - shown on hover */}
        <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => onEdit(diary, e)}
            className="w-8 h-8 flex items-center justify-center rounded-lg bg-amber-100/70 dark:bg-amber-950/30 text-amber-600 dark:text-amber-400 hover:text-amber-800 dark:hover:text-amber-300 hover:bg-amber-200/70 dark:hover:bg-amber-900/40 transition-all"
            title="编辑"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
            </svg>
          </button>
          <button
            onClick={(e) => onDelete(diary, e)}
            className="w-8 h-8 flex items-center justify-center rounded-lg bg-amber-100/70 dark:bg-amber-950/30 text-amber-600 dark:text-amber-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-100/70 dark:hover:bg-rose-950/30 transition-all"
            title="删除"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>

        {/* Date */}
        <div className="flex justify-between items-start mb-3 pr-20">
          <h3 className="font-semibold text-amber-900 dark:text-amber-100 text-base flex items-center gap-2">
            <span className="text-xl">📔</span>
            {dateStr}
          </h3>
          {tag && (
            <span className="inline-flex items-center px-2.5 py-1 bg-amber-200/70 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 text-xs font-medium rounded-full">
              {tag}
            </span>
          )}
        </div>

        {/* Content preview */}
        <p className="text-amber-800/80 dark:text-amber-200/80 text-sm leading-relaxed line-clamp-3 whitespace-pre-wrap">
          {displayContent}
        </p>
      </div>
    </div>
  );
};
