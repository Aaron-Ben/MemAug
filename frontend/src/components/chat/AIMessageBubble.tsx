/** AI Message Bubble Component - 独立的AI消息气泡组件，便于后续美化 */

import React, { useState } from 'react';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export interface AIMessageBubbleProps {
  /** 消息内容 */
  content: string;
  /** 是否正在流式输出 */
  isStreaming?: boolean;
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
 * 工具请求可折叠组件
 */
const ToolRequestCollapsible: React.FC<{ content: string }> = ({ content }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // 解析工具请求内容
  const parseToolRequest = (text: string) => {
    const params: Record<string, string> = {};
    // 匹配 tool_name, maid, keyword, windowsize 等参数
    const matches = text.match(/(\w+):「始」(.+?)「末」/g);
    if (matches) {
      matches.forEach((match) => {
        const [, key, value] = match.match(/(\w+):「始」(.+?)「末」/) || [];
        if (key && value) {
          params[key] = value;
        }
      });
    }
    return params;
  };

  const params = parseToolRequest(content);

  return (
    <div className="my-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between gap-2 px-4 py-2.5 bg-blue-50 dark:bg-blue-950/30 border-2 border-blue-200 dark:border-blue-800 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors cursor-pointer group"
      >
        <div className="flex items-center gap-2">
          <svg
            className={clsx('w-4 h-4 text-blue-600 dark:text-blue-400 transition-transform duration-200', isExpanded && 'rotate-90')}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
            🔧 工具调用: {params.tool_name || 'Unknown'}
          </span>
        </div>
        <span className={clsx('text-xs text-blue-500 dark:text-blue-400 transition-transform duration-200', isExpanded && 'rotate-180')}>
          ▼
        </span>
      </button>

      {isExpanded && (
        <div className="mt-2 px-4 py-3 bg-blue-50/50 dark:bg-blue-950/20 border-2 border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="space-y-1.5 text-sm">
            {Object.entries(params).map(([key, value]) => (
              <div key={key} className="flex gap-2">
                <span className="font-semibold text-blue-700 dark:text-blue-400 min-w-[80px]">
                  {key}:
                </span>
                <span className="text-neutral-700 dark:text-neutral-300 break-all">
                  {value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * AI消息气泡组件 - 专门用于显示AI的回复
 *
 * 这个组件可以独立进行样式美化，包括：
 * - 背景渐变效果
 * - 边框样式
 * - 阴影效果
 * - 动画效果
 * 等等
 */
export const AIMessageBubble: React.FC<AIMessageBubbleProps> = ({
  content,
  isStreaming = false,
  timestamp,
  className,
}) => {
  // 解析消息内容，提取工具请求
  const parseContent = (text: string) => {
    const toolRequestRegex = /<<<\[TOOL_REQUEST\]>>>(.+?)<<<\[END_TOOL_REQUEST\]>>>/gs;
    const matches = Array.from(text.matchAll(toolRequestRegex));

    if (matches.length === 0) {
      return { toolRequests: [], content: text };
    }

    const toolRequests = matches.map(match => match[1].trim());
    let cleanContent = text.replace(toolRequestRegex, '').trim();

    return { toolRequests, content: cleanContent };
  };

  const { toolRequests, content: cleanContent } = parseContent(content);

  return (
    <div className={clsx('flex flex-col max-w-[85%] md:max-w-[70%] items-start', className)}>
      {/* AI消息气泡主体 */}
      <div className="bg-white dark:bg-neutral-800 text-neutral-800 dark:text-neutral-100 rounded-2xl rounded-bl-sm px-5 py-3 shadow-sm border border-neutral-100 dark:border-neutral-700">
        <div className="text-base leading-relaxed break-words markdown-content">
          {/* 工具请求折叠框 */}
          {toolRequests.map((toolRequest, index) => (
            <ToolRequestCollapsible key={index} content={toolRequest} />
          ))}

          {/* 正常的 Markdown 内容 */}
          {cleanContent && (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // 标题样式
                h1: ({ children }) => <h1 className="text-xl font-bold mt-3 mb-2">{children}</h1>,
                h2: ({ children }) => <h2 className="text-lg font-bold mt-2 mb-1">{children}</h2>,
                h3: ({ children }) => <h3 className="text-base font-semibold mt-2 mb-1">{children}</h3>,
                // 段落样式
                p: ({ children }) => <p className="my-1">{children}</p>,
                // 列表样式
                ul: ({ children }) => <ul className="list-disc list-inside my-1 ml-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside my-1 ml-2">{children}</ol>,
                li: ({ children }) => <li className="my-0.5">{children}</li>,
                // 代码块样式
                code: ({ className, children }) => {
                  const isInline = !className;
                  return isInline ? (
                    <code className="bg-neutral-100 dark:bg-neutral-700 px-1.5 py-0.5 rounded text-sm font-mono text-rose-600 dark:text-rose-400">
                      {children}
                    </code>
                  ) : (
                    <code className={clsx('block bg-neutral-100 dark:bg-neutral-700 px-3 py-2 rounded-lg text-sm font-mono overflow-x-auto my-2', className)}>
                      {children}
                    </code>
                  );
                },
                pre: ({ children }) => <pre className="bg-neutral-100 dark:bg-neutral-700 rounded-lg p-3 overflow-x-auto my-2">{children}</pre>,
                // 引用样式
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-neutral-300 dark:border-neutral-600 pl-3 italic my-2 text-neutral-600 dark:text-neutral-400">
                    {children}
                  </blockquote>
                ),
                // 分隔线样式
                hr: () => <hr className="border-neutral-200 dark:border-neutral-700 my-3" />,
                // 链接样式
                a: ({ href, children }) => (
                  <a href={href} className="text-rose-500 dark:text-rose-400 hover:underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                // 加粗和斜体
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
              }}
            >
              {cleanContent}
            </ReactMarkdown>
          )}
        </div>

        {/* 流式输出时的指示器 */}
        {isStreaming && (
          <div className="flex gap-1 mt-2">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 dark:bg-rose-500 animate-pulse-subtle"></span>
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 dark:bg-rose-500 animate-pulse-subtle delay-150"></span>
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 dark:bg-rose-500 animate-pulse-subtle delay-225"></span>
          </div>
        )}
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

/**
 * AI加载状态组件
 */
export const AILoadingBubble: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={clsx('flex flex-col max-w-[85%] md:max-w-[70%] items-start', className)}>
      <div className="bg-white dark:bg-neutral-800 rounded-2xl rounded-bl-sm px-5 py-3 shadow-sm border border-neutral-100 dark:border-neutral-700 min-w-[60px]">
        <div className="flex gap-1.5 items-center">
          <span className="w-2 h-2 rounded-full bg-rose-400 dark:bg-rose-500 animate-typing"></span>
          <span className="w-2 h-2 rounded-full bg-rose-400 dark:bg-rose-500 animate-typing delay-150"></span>
          <span className="w-2 h-2 rounded-full bg-rose-400 dark:bg-rose-500 animate-typing delay-225"></span>
        </div>
      </div>
    </div>
  );
};
