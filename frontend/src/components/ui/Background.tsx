/** Background Component - 智能背景组件，使用 filter 方案处理暗色模式 */

import React from 'react';
import { cn } from '../../utils';
import { useTheme } from '../../contexts';

export interface BackgroundProps {
  /** 背景图片 URL */
  imageUrl?: string;
  /** 自定义类名 */
  className?: string;
  /** 背景渐变色（亮色模式） */
  lightGradient?: string;
  /** 背景渐变色（暗色模式） */
  darkGradient?: string;
  /** 是否显示纹理 */
  showTexture?: boolean;
  /** 子元素（如覆盖在背景上的内容） */
  children?: React.ReactNode;
}

/**
 * 背景组件 - 使用 CSS filter 处理暗色模式
 *
 * 暗色模式处理方案：
 * - brightness(0.7) - 降低亮度到 70%
 * - grayscale(40%) - 添加 40% 灰度
 * - contrast(0.9) - 稍微降低对比度
 *
 * 优点：
 * - 性能好，使用原生 CSS 滤镜
 * - 实现简单，不需要额外的 DOM 元素
 * - 过渡平滑，视觉效果自然
 */
export const Background: React.FC<BackgroundProps> = ({
  imageUrl,
  className,
  lightGradient = 'from-rose-50/30 via-white to-pink-50/20',
  darkGradient = 'from-night-primary via-night-secondary to-night-primary',
  showTexture = true,
  children,
}) => {
  const { isDark } = useTheme();

  return (
    <div className={cn('relative h-full w-full', className)}>
      {/* 基础渐变背景 */}
      <div className={cn(
        'absolute inset-0 bg-gradient-to-br transition-all duration-700',
        lightGradient,
        isDark && darkGradient
      )} />

      {/* 背景图片层 */}
      {imageUrl && (
        <div className="absolute inset-0">
          {/* 亮色模式背景图 */}
          <div
            className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-40 transition-opacity duration-700"
            style={{ backgroundImage: `url(${imageUrl})` }}
          />

          {/* 暗色模式滤镜层 */}
          <div
            className={cn(
              'absolute inset-0 bg-cover bg-center bg-no-repeat transition-opacity duration-700',
              'opacity-0 dark:opacity-35'
            )}
            style={{
              backgroundImage: `url(${imageUrl})`,
              filter: 'brightness(0.7) grayscale(40%) contrast(0.9)',
            }}
          />
        </div>
      )}

      {/* 微妙的纹理层 */}
      {showTexture && (
        <div
          className={cn(
            'absolute inset-0 pointer-events-none transition-opacity duration-700',
            'opacity-[0.03] dark:opacity-[0.05]'
          )}
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
          }}
        />
      )}

      {/* 光晕效果 - 仅在暗色模式下显示 */}
      {isDark && (
        <>
          <div className="absolute top-0 left-0 w-96 h-96 bg-rose-500/5 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-pink-500/5 rounded-full blur-3xl pointer-events-none" />
        </>
      )}

      {/* 子元素 */}
      {children}
    </div>
  );
};

/**
 * 简化版背景组件 - 用于简单的背景需求
 */
export const SimpleBackground: React.FC<{
  imageUrl?: string;
  className?: string;
  children?: React.ReactNode;
}> = ({ imageUrl, className, children }) => {
  return (
    <div className={cn('relative h-full w-full', className)}>
      {/* 渐变背景 */}
      <div className={cn(
        'absolute inset-0 bg-gradient-to-br transition-all duration-500',
        'from-rose-50/30 via-white to-pink-50/20',
        'dark:from-night-primary dark:via-night-secondary dark:to-night-primary'
      )} />

      {/* 背景图 */}
      {imageUrl && (
        <div
          className={cn(
            'absolute inset-0 bg-cover bg-center bg-no-repeat transition-all duration-500',
            'opacity-40 dark:opacity-25'
          )}
          style={{
            backgroundImage: `url(${imageUrl})`,
            filter: 'var(--tw-brightness, 1)',
          }}
        />
      )}

      {/* 暗色叠加层 */}
      <div className="absolute inset-0 bg-gradient-to-t from-rose-950/5 dark:from-night-primary/30 pointer-events-none transition-all duration-500" />

      {children}
    </div>
  );
};

