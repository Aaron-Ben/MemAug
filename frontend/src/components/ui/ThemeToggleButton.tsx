/** Theme Toggle Button - 精美的主题切换按钮 */

import React, { useState } from 'react';
import { useTheme, type ThemeMode } from '../../contexts/ThemeContext';
import { cn } from '../../utils';

export interface ThemeToggleButtonProps {
  /** 自定义类名 */
  className?: string;
  /** 尺寸变体 */
  size?: 'sm' | 'md' | 'lg';
  /** 显示文字标签 */
  showLabel?: boolean;
  /** 样式变体 */
  variant?: 'floating' | 'inline' | 'minimal';
}

const themes: { mode: ThemeMode; icon: string; label: string }[] = [
  { mode: 'light', icon: '☀️', label: '亮色' },
  { mode: 'dark', icon: '🌙', label: '暗色' },
  { mode: 'auto', icon: '🌓', label: '自动' },
];

/**
 * 主题切换按钮组件
 *
 * 支持三种样式变体：
 * - floating: 悬浮按钮样式（带阴影和悬浮效果）
 * - inline: 内联按钮样式（简洁的胶囊形状）
 * - minimal: 极简样式（仅图标）
 */
export const ThemeToggleButton: React.FC<ThemeToggleButtonProps> = ({
  className,
  size = 'md',
  showLabel = false,
  variant = 'floating',
}) => {
  const { theme, setTheme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  const currentIndex = themes.findIndex(t => t.mode === theme);
  const currentTheme = themes[currentIndex];

  const handleCycleTheme = () => {
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex].mode);
  };

  // 尺寸样式
  const sizeStyles = {
    sm: 'h-8 px-2 text-sm',
    md: 'h-10 px-3 text-base',
    lg: 'h-12 px-4 text-lg',
  };

  const iconSize = {
    sm: 'text-base',
    md: 'text-lg',
    lg: 'text-xl',
  };

  // 变体样式
  const variantStyles = {
    floating: cn(
      'bg-white/90 dark:bg-neutral-800/90 backdrop-blur-md',
      'border-2 border-rose-200 dark:border-rose-800/50',
      'shadow-lg shadow-rose-200/50 dark:shadow-neutral-900/50',
      'hover:shadow-xl hover:shadow-rose-300/50 dark:hover:shadow-neutral-900/70',
      'hover:-translate-y-0.5 active:translate-y-0',
      'transition-all duration-300'
    ),
    inline: cn(
      'bg-gradient-to-r from-rose-100 to-pink-100 dark:from-neutral-800 dark:to-neutral-700',
      'border border-rose-200 dark:border-neutral-600',
      'hover:from-rose-200 hover:to-pink-200 dark:hover:from-neutral-700 dark:hover:to-neutral-600',
      'transition-all duration-200'
    ),
    minimal: cn(
      'bg-transparent',
      'hover:bg-rose-50 dark:hover:bg-neutral-800/50',
      'transition-colors duration-200'
    ),
  };

  if (variant === 'minimal') {
    return (
      <button
        onClick={handleCycleTheme}
        className={cn(
          'rounded-lg flex items-center gap-2',
          sizeStyles[size],
          variantStyles[variant],
          className
        )}
        title={`当前主题: ${currentTheme.label} (点击切换)`}
        type="button"
      >
        <span className={iconSize[size]}>{currentTheme.icon}</span>
        {showLabel && (
          <span className="text-sm font-medium text-neutral-700 dark:text-neutral-200">
            {currentTheme.label}
          </span>
        )}
      </button>
    );
  }

  if (variant === 'inline') {
    return (
      <button
        onClick={handleCycleTheme}
        className={cn(
          'rounded-full flex items-center gap-2',
          sizeStyles[size],
          variantStyles[variant],
          className
        )}
        title={`当前主题: ${currentTheme.label} (点击切换)`}
        type="button"
      >
        <span className={iconSize[size]}>{currentTheme.icon}</span>
        {showLabel && (
          <span className="text-sm font-semibold text-neutral-700 dark:text-neutral-200">
            {currentTheme.label}
          </span>
        )}
      </button>
    );
  }

  // Floating variant - 更丰富的交互
  return (
    <div className="relative">
      {/* 主按钮 */}
      <button
        onClick={handleCycleTheme}
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        className={cn(
          'rounded-full flex items-center gap-2',
          sizeStyles[size],
          variantStyles[variant],
          'cursor-pointer',
          className
        )}
        type="button"
      >
        <span className={cn(iconSize[size], 'transition-transform duration-300', isOpen && 'rotate-12')}>
          {currentTheme.icon}
        </span>
        {showLabel && (
          <span className="text-sm font-semibold text-neutral-700 dark:text-neutral-200">
            {currentTheme.label}
          </span>
        )}
      </button>

      {/* 悬浮时显示的主题预览 */}
      {isOpen && (
        <div
          className={cn(
            'absolute top-full mt-2 left-1/2 -translate-x-1/2',
            'bg-white/95 dark:bg-neutral-800/95 backdrop-blur-md',
            'rounded-2xl shadow-xl shadow-rose-200/50 dark:shadow-neutral-900/70',
            'border border-rose-100 dark:border-neutral-700',
            'p-1.5 gap-1 flex',
            'animate-in fade-in slide-in-from-top-1 duration-200',
            'z-50'
          )}
          onMouseEnter={() => setIsOpen(true)}
          onMouseLeave={() => setIsOpen(false)}
        >
          {themes.map((t) => (
            <button
              key={t.mode}
              onClick={() => setTheme(t.mode)}
              className={cn(
                'w-10 h-10 rounded-xl flex items-center justify-center text-lg',
                'transition-all duration-200',
                'hover:scale-110 active:scale-95',
                theme === t.mode
                  ? 'bg-gradient-to-br from-rose-400 to-pink-500 text-white shadow-md'
                  : 'hover:bg-rose-50 dark:hover:bg-neutral-700'
              )}
              title={t.label}
              type="button"
            >
              {t.icon}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * 简化的主题切换按钮（用于工具栏等紧凑空间）
 */
export const CompactThemeToggle: React.FC<{ className?: string }> = ({ className }) => {
  const { theme, setTheme, isDark } = useTheme();

  const cycleTheme = () => {
    const themes: ThemeMode[] = ['light', 'dark', 'auto'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    setTheme(nextTheme);
  };

  return (
    <button
      onClick={cycleTheme}
      className={cn(
        'w-9 h-9 rounded-lg',
        'flex items-center justify-center text-base',
        'bg-gradient-to-br from-rose-100 to-pink-100 dark:from-neutral-700 dark:to-neutral-600',
        'border border-rose-200 dark:border-neutral-500',
        'hover:from-rose-200 hover:to-pink-200 dark:hover:from-neutral-600 dark:hover:to-neutral-500',
        'transition-all duration-200',
        'shadow-sm hover:shadow-md',
        className
      )}
      title={`主题: ${theme === 'auto' ? (isDark ? '暗色(自动)' : '亮色(自动)') : theme === 'dark' ? '暗色' : '亮色'}`}
      type="button"
    >
      {theme === 'auto' ? (
        isDark ? '🌙' : '☀️'
      ) : theme === 'dark' ? (
        '🌙'
      ) : (
        '☀️'
      )}
    </button>
  );
};
