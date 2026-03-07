/** Diary detail modal component - Refined elegant style */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { DiaryEntry, extractDateFromPath } from '../../services/diaryService';

interface DiaryDetailModalProps {
  diary: DiaryEntry | null;
  isOpen: boolean;
  onClose: () => void;
  characterName?: string;
}

/**
 * 将 LaTeX 数学公式格式转换为 remark-math 格式
 */
const convertLatexMath = (text: string): string => {
  let result = text;

  // 处理块级公式 \[ ... \] -> $$ ... $$
  result = result.replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_match, content) => {
    return `$$${content.trim()}$$`;
  });

  // 处理行内公式 \( ... \) -> $ ... $
  result = result.replace(/\\\(\s*(.*?)\s*\\\)/g, (_match, content) => {
    return `$${content.trim()}$`;
  });

  return result;
};

export const DiaryDetailModal: React.FC<DiaryDetailModalProps> = ({
  diary,
  isOpen,
  onClose,
  characterName
}) => {
  if (!isOpen || !diary) return null;

  // Extract date from path
  const date = extractDateFromPath(diary.path);

  // 转换 LaTeX 数学公式格式
  const formattedContent = convertLatexMath(diary.content);

  return (
    <div
      className="fixed inset-0 bg-black/20 backdrop-blur-sm animate-fade-in z-50"
      onClick={onClose}
    >
      <div
        className="mx-4 my-8 max-w-2xl bg-amber-50/95 dark:bg-stone-800/95 rounded-3xl shadow-xl border border-amber-100 dark:border-stone-600 max-h-[calc(100vh-4rem)] flex flex-col animate-message-in overflow-hidden backdrop-blur-sm"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-amber-100/80 via-rose-50/60 to-amber-50/50 dark:from-stone-700/50 dark:via-rose-950/20 dark:to-stone-800/50 px-6 py-5 border-b border-amber-200/50 dark:border-stone-600">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl">📔</span>
                <h2 className="text-2xl font-bold text-amber-900 dark:text-amber-100">{characterName ? `${characterName}的日记` : '日记'}</h2>
              </div>
              <p className="text-sm text-amber-700/80 dark:text-amber-300/70">
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
          {/* Diary content with Markdown support */}
          <div className="mb-6 markdown-content">
            <ReactMarkdown
              remarkPlugins={[[remarkMath, { singleDollarTextMath: true }], remarkGfm]}
              rehypePlugins={[rehypeKatex]}
            >
              {formattedContent}
            </ReactMarkdown>
          </div>

          {/* Metadata */}
          <div className="pt-4 border-t border-amber-200/50 dark:border-stone-600 text-xs text-amber-600/70 dark:text-amber-400/60 space-y-1">
            <p>角色 ID: {diary.character_id}</p>
            <p>文件路径: {diary.path}</p>
            <p>修改时间: {new Date(diary.mtime * 1000).toLocaleString('zh-CN')}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
