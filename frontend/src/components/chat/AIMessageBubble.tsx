/** AI Message Bubble Component - 独立的AI消息气泡组件，便于后续美化 */

import React, { useState } from 'react';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { cn } from '../../utils';

export interface AIMessageBubbleProps {
  /** 消息内容 */
  content: string;
  /** 是否正在流式输出 */
  isStreaming?: boolean;
  /** 消息时间（可选） */
  timestamp?: Date;
  /** 自定义类名 */
  className?: string;
  /** 角色 ID（可选，用于替换工具调用中的占位符） */
  characterId?: string;
  /** 角色名称（可选，用于替换工具调用中的占位符） */
  characterName?: string;
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
 * 检查是否为日记相关工具
 */
const isDiaryTool = (toolName: string): boolean => {
  return toolName.toLowerCase().includes('daily') ||
         toolName.toLowerCase().includes('diary') ||
         toolName.toLowerCase().includes('日记');
};

/**
 * 工具请求可折叠组件 - 鼠标悬停自动展开
 */
const ToolRequestCollapsible: React.FC<{ content: string; characterId?: string; characterName?: string }> = ({ content, characterId, characterName }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // 解析工具请求内容
  const parseToolRequest = (text: string) => {
    const params: Record<string, string> = {};
    // 匹配 tool_name, maid, keyword, windowsize, Content 等参数
    // 使用 [\s\S]+? 来匹配包括换行符在内的所有字符
    const matches = text.match(/(\w+):「始」([\s\S]+?)「末」/g);
    if (matches) {
      matches.forEach((match) => {
        const [, key, value] = match.match(/(\w+):「始」([\s\S]+?)「末」/) || [];
        if (key && value) {
          // 替换占位符为实际值
          let processedValue = value;
          if (characterId) processedValue = processedValue.replace(/\{CHARACTER_ID\}/g, characterId);
          if (characterName) processedValue = processedValue.replace(/\{CHARACTER_NAME\}/g, characterName);
          // 替换日期占位符
          processedValue = processedValue.replace(/\{TODAY\}/g, new Date().toISOString().split('T')[0]);
          processedValue = processedValue.replace(/\{CURRENT_TIME\}/g, new Date().toTimeString().slice(0, 5));

          params[key] = processedValue;
        }
      });
    }
    return params;
  };

  const params = parseToolRequest(content);
  const toolName = params.tool_name || 'Unknown';
  const isDiary = isDiaryTool(toolName);
  const todayDate = new Date().toISOString().split('T')[0];


  // 获取显示用的角色名称
  const displayMaidName = characterName;

  return (
    <div
      className="my-3 group"
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      {/* 大框：根据工具类型使用不同样式 */}
      <div className={cn(
        "rounded-lg p-4 transition-all duration-300 border",
        isDiary
          ? "bg-amber-50/95 dark:bg-stone-800/95 text-amber-900 dark:text-amber-100 border-amber-200/50 dark:border-stone-600"
          : "bg-slate-100/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 border-slate-300 dark:border-slate-600"
      )}>
        {/* 折叠状态的标题栏 */}
        <div className="flex items-center justify-between gap-2 cursor-pointer">
          <div className="flex items-center gap-2">
            <svg
              className={cn('w-4 h-4 transition-transform duration-200', isExpanded && 'rotate-90')}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className={cn("font-medium flex items-center gap-1.5", isDiary ? "text-sm" : "text-sm")}>
              {isDiary ? (
                <><span className="text-base">📔</span>日记记录</>
              ) : (
                <><span className="text-base">⚙️</span><span className="text-slate-600 dark:text-slate-300">工具调用</span>: <span className="text-sky-600 dark:text-sky-400 font-mono text-xs">{toolName}</span></>
              )}
            </span>
          </div>
          <span className={cn('text-xs transition-transform duration-200', isExpanded && 'rotate-180')}>
            ▼
          </span>
        </div>

        {/* 展开后的内容区域 */}
        {isExpanded && (
          <div className="mt-4 animate-in fade-in slide-in-from-top-2 duration-200">
            {/* 日记特殊样式：显示日记头部 */}
            {isDiary && (
              <>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 border-b border-amber-200/50 dark:border-stone-600 pb-2">
                  <h2 className="text-xl font-serif font-bold text-amber-900 dark:text-amber-100">{displayMaidName} Diary</h2>
                  <span className="text-sm text-amber-600/70 dark:text-amber-400/60">{todayDate}</span>
                </div>
                {params.maid && (
                  <div className="mb-4">
                    <span className="text-amber-600/70 dark:text-amber-400/60 font-medium">Maid:</span>
                    <span className="ml-2 px-2 py-0.5 bg-amber-100/70 dark:bg-amber-900/40 rounded text-amber-800 dark:text-amber-200 font-medium">
                      {displayMaidName}
                    </span>
                  </div>
                )}
              </>
            )}

            {/* 内容区域 */}
            <div className={cn(
              "rounded-md p-4 space-y-4 max-h-96 overflow-y-auto",
              isDiary
                ? "bg-amber-100/60 dark:bg-stone-700/50 text-amber-900/90 dark:text-amber-100/90 text-sm"
                : "bg-slate-50/80 dark:bg-slate-900/50 text-slate-700 dark:text-slate-300 p-3 space-y-2 text-sm"
            )}>
              {Object.entries(params).map(([key, value]) => {
                // 过滤掉已经在头部显示的字段
                if (isDiary && (key === 'tool_name' || key === 'maid' || key === 'Date')) return null;

                return (
                  <div key={key} className={isDiary ? "space-y-1" : "flex gap-2"}>
                    {isDiary ? (
                      // 日记风格：直接显示内容
                      <ReactMarkdown
                        remarkPlugins={[[remarkMath, { singleDollarTextMath: true }], remarkGfm]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                          p: ({ node, ...props }) => <p className="mb-2 leading-relaxed" {...props} />,
                          strong: ({ node, ...props }) => <span className="text-amber-700 dark:text-amber-300 font-semibold" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />,
                          li: ({ node, ...props }) => <li className="mb-1" {...props} />,
                        }}
                      >
                        {value}
                      </ReactMarkdown>
                    ) : (
                      // 普通工具风格：键值对显示
                      <>
                        <span className="font-semibold min-w-[80px] flex-shrink-0 text-sm text-slate-500 dark:text-slate-400">
                          {key}:
                        </span>
                        <div className="break-words flex-1 text-sm prose prose-sm max-w-none prose-headings:text-slate-700 prose-p:text-slate-600 prose-strong:text-slate-800 prose-code:text-sky-600 dark:prose-headings:text-slate-200 dark:prose-p:text-slate-300 dark:prose-strong:text-slate-100 dark:prose-code:text-sky-400">
                          <ReactMarkdown
                            remarkPlugins={[[remarkMath, { singleDollarTextMath: true }], remarkGfm]}
                            rehypePlugins={[rehypeKatex]}
                          >
                            {value}
                          </ReactMarkdown>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * 将 LaTeX 数学公式格式转换为 remark-math 格式
 * LaTeX 标准格式：
 * - \( ... \) 表示行内公式
 * - \[ ... \] 表示块级公式
 *
 * 转换为：
 * - $ ... $ 表示行内公式
 * - $$ ... $$ 表示块级公式
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
  characterId,
  characterName,
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
  // 转换 LaTeX 数学公式格式
  const formattedContent = convertLatexMath(cleanContent);

  return (
    <div className={clsx('flex flex-col max-w-[85%] md:max-w-[70%] items-start', className)}>
      {/* AI消息气泡主体 - 精美的玻璃态效果 */}
      <div className="relative group">
        {/* 背景光晕效果 */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-200 to-pink-200 dark:from-rose-950 dark:to-pink-950 rounded-2xl rounded-bl-sm opacity-0 group-hover:opacity-30 transition-opacity duration-300 blur-md"></div>

        <div className="relative bg-white/80 dark:bg-night-elevated/80 backdrop-blur-sm text-neutral-800 dark:text-neutral-100 rounded-2xl rounded-bl-sm px-5 py-3 shadow-soft dark:shadow-dark-soft border border-neutral-100/50 dark:border-neutral-700/50 hover:shadow-soft-lg dark:hover:shadow-dark-soft-lg transition-all duration-300">
          <div className="text-base leading-relaxed break-words markdown-content">
            {/* 工具请求折叠框 */}
            {toolRequests.map((toolRequest, index) => (
              <ToolRequestCollapsible
                key={index}
                content={toolRequest}
                characterId={characterId}
                characterName={characterName}
              />
            ))}

            {/* 正常的 Markdown 内容 */}
            {cleanContent && (
              <ReactMarkdown
                remarkPlugins={[[remarkMath, { singleDollarTextMath: true }], remarkGfm]}
                rehypePlugins={[rehypeKatex]}
              >
                {formattedContent}
              </ReactMarkdown>
            )}
          </div>

          {/* 流式输出时的指示器 - 精美的动画效果 */}
          {isStreaming && (
            <div className="flex gap-1.5 mt-3">
              <span className="w-2 h-2 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-pulse-subtle"></span>
              <span className="w-2 h-2 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-pulse-subtle delay-150"></span>
              <span className="w-2 h-2 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-pulse-subtle delay-300"></span>
            </div>
          )}
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

/**
 * AI加载状态组件 - 精美的加载动画
 */
export const AILoadingBubble: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={clsx('flex flex-col max-w-[85%] md:max-w-[70%] items-start', className)}>
      <div className="relative">
        {/* 发光效果 */}
        <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-200 to-pink-200 dark:from-rose-950 dark:to-pink-950 rounded-2xl rounded-bl-sm opacity-30 animate-pulse-subtle blur-md"></div>

        <div className="relative bg-white/80 dark:bg-night-elevated/80 backdrop-blur-sm rounded-2xl rounded-bl-sm px-5 py-3 shadow-soft dark:shadow-dark-soft border border-neutral-100/50 dark:border-neutral-700/50 min-w-[60px]">
          <div className="flex gap-2 items-center">
            <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-typing"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-typing delay-150"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-r from-rose-400 to-pink-500 animate-typing delay-225"></span>
          </div>
        </div>
      </div>
    </div>
  );
};
