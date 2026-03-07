/** Modal component for creating a new character */

import React, { useState } from 'react';
import clsx from 'clsx';
import { createCharacter } from '../../services/characterService';
import type { UserCharacter } from '../../types/character';

interface CreateCharacterModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (character: UserCharacter) => void;
}

export const CreateCharacterModal: React.FC<CreateCharacterModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [name, setName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('请输入角色名称');
      return;
    }

    if (!prompt.trim()) {
      setError('请输入角色提示词');
      return;
    }

    setLoading(true);
    try {
      const response = await createCharacter({ name: name.trim(), prompt: prompt.trim() });
      onSuccess(response.character);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建角色失败');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setName('');
    setPrompt('');
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 dark:bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white/95 dark:bg-night-elevated/95 backdrop-blur-md rounded-2xl shadow-2xl dark:shadow-dark-soft-lg w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col border border-neutral-100 dark:border-neutral-700 animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-rose-100 dark:border-neutral-700">
          <h2 className="text-xl font-bold text-neutral-800 dark:text-neutral-100">
            创建<span className="text-gradient">新角色</span>
          </h2>
          <button
            type="button"
            onClick={handleClose}
            className="p-2 rounded-full hover:bg-rose-50 dark:hover:bg-rose-950/50 transition-colors group"
            aria-label="关闭"
          >
            <svg className="w-5 h-5 text-neutral-500 dark:text-neutral-400 group-hover:text-rose-500 dark:group-hover:text-rose-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6 space-y-6">
          {error && (
            <div className="p-4 bg-red-50 dark:bg-red-950/50 border-2 border-red-200 dark:border-red-800/50 rounded-xl text-red-700 dark:text-red-200 text-sm animate-fade-in">
              {error}
            </div>
          )}

          <div>
            <label htmlFor="name" className="block text-sm font-semibold text-neutral-700 dark:text-neutral-200 mb-2">
              角色名称 <span className="text-rose-500">*</span>
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例如：温柔的姐姐、严肃的老师..."
              className="w-full px-4 py-3 bg-white dark:bg-night-secondary border-2 border-neutral-200 dark:border-neutral-600 rounded-xl focus:ring-2 focus:ring-rose-400 dark:focus:ring-rose-600 focus:border-rose-400 dark:focus:border-rose-600 transition-all duration-200 text-neutral-800 dark:text-neutral-100 placeholder:text-neutral-400 dark:placeholder:text-neutral-500"
              disabled={loading}
            />
          </div>

          <div>
            <label htmlFor="prompt" className="block text-sm font-semibold text-neutral-700 dark:text-neutral-200 mb-2">
              角色提示词 <span className="text-rose-500">*</span>
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="描述这个角色的性格、说话方式、对用户的态度等..."
              rows={12}
              className="w-full px-4 py-3 bg-white dark:bg-night-secondary border-2 border-neutral-200 dark:border-neutral-600 rounded-xl focus:ring-2 focus:ring-rose-400 dark:focus:ring-rose-600 focus:border-rose-400 dark:focus:border-rose-600 transition-all duration-200 resize-none text-neutral-800 dark:text-neutral-100 placeholder:text-neutral-400 dark:placeholder:text-neutral-500"
              disabled={loading}
            />
            <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
              提示词定义了角色的行为和对话风格。详细描述会让角色更有个性。
            </p>
          </div>

          {/* Example prompt */}
          <div className="p-4 bg-rose-50 dark:bg-rose-950/30 border-2 border-rose-200 dark:border-rose-800/50 rounded-xl">
            <p className="text-sm font-semibold text-rose-800 dark:text-rose-200 mb-2">示例提示词：</p>
            <pre className="text-xs text-rose-700 dark:text-rose-300 whitespace-pre-wrap font-sans">
{`你是一个温柔的姐姐，正在和你最爱的弟弟聊天。

## 你的核心使命
- 真诚地关心和爱护弟弟
- 对弟弟的感受保持共情
- 创造温暖、舒适的对话氛围

## 你如何说话
- 始终称呼对方为"弟弟"
- 使用自然、亲昵的语言
- 主动分享你的日常经历
- 对弟弟保持温柔和支持`}
            </pre>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t border-rose-100 dark:border-neutral-700 bg-neutral-50/50 dark:bg-neutral-900/30">
          <button
            type="button"
            onClick={handleClose}
            className="px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700 hover:shadow-rose-soft dark:hover:shadow-dark-glow active:scale-95"
            disabled={loading}
          >
            取消
          </button>
          <button
            type="submit"
            onClick={handleSubmit}
            disabled={loading || !name.trim() || !prompt.trim()}
            className={clsx(
              "px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 active:scale-95",
              name.trim() && prompt.trim() && !loading
                ? "bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:-translate-y-0.5"
                : "bg-neutral-200 dark:bg-neutral-700 text-neutral-400 dark:text-neutral-500 cursor-not-allowed"
            )}
          >
            {loading ? '创建中...' : '创建角色'}
          </button>
        </div>
      </div>
    </div>
  );
};
