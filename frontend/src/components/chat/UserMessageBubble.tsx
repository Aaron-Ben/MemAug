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
      {/* 用户消息气泡主体 */}
      <div className="bg-gradient-to-br from-rose-400 to-rose-500 text-white rounded-2xl rounded-br-sm px-5 py-3 shadow-sm">
        <div className="text-base leading-relaxed break-words">
          {content}
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
