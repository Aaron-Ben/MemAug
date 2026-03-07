/** User Message Bubble Component - 独立的用户消息气泡组件，便于后续美化 */

import React from 'react';
import { clsx } from 'clsx';

export interface UserMessageBubbleProps {
  /** 消息内容 */
  content: string;
  /** 消息时间（可选） */
  timestamp?: Date;
  /** 自定义类名 */
  className?: string;
}

/**
 * 格式化消息时间为相对时间
 */
const formatMessageTime = (date: Date) => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);

  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes}分钟前`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}小时前`;

  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}天前`;

  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
};

/**
 * 用户消息气泡组件 - 专门用于显示用户的输入
 */
export const UserMessageBubble: React.FC<UserMessageBubbleProps> = ({
  content,
  timestamp,
  className,
}) => {
  return (
    <div className={clsx('flex flex-col max-w-[85%] md:max-w-[70%] items-end', className)}>
      {/* 用户消息气泡主体 - 精美渐变效果 */}
      <div className="relative group">
        {/* 发光效果 */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-400 to-pink-500 rounded-2xl rounded-br-sm opacity-0 group-hover:opacity-20 transition-opacity duration-300 blur-sm"></div>

        <div className="relative bg-gradient-to-br from-rose-400 via-rose-500 to-pink-500 text-white rounded-2xl rounded-br-sm px-5 py-3 shadow-rose-soft hover:shadow-rose-soft-lg transition-all duration-300">
          <div className="text-base leading-relaxed break-words font-medium">
            {content}
          </div>

          {/* 微妙的高光效果 */}
          <div className="absolute inset-0 rounded-2xl rounded-br-sm bg-gradient-to-br from-white/20 to-transparent pointer-events-none"></div>
        </div>
      </div>

      {/* 消息时间戳 */}
      {timestamp && (
        <div className="text-[11px] text-neutral-400 dark:text-neutral-500 mt-1 px-1">
          {formatMessageTime(timestamp)}
        </div>
      )}
    </div>
  );
};
