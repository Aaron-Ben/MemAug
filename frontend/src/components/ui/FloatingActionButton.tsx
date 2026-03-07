/** Floating Action Button component - Refined elegant style */

import React from 'react';
import { clsx } from 'clsx';

interface FloatingActionButtonProps {
  onClick: () => void;
  icon: React.ReactNode;
  ariaLabel: string;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  index?: number; // For spacing multiple buttons
}

export const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  onClick,
  icon,
  ariaLabel,
  position = 'bottom-right',
  index = 0,
}) => {
  const getPositionStyle = (pos: typeof position, idx: number) => {
    const baseSpacing = 16; // 1rem
    const buttonSpacing = 64; // 56px button + 8px gap
    const spacing = baseSpacing + (idx * buttonSpacing);

    switch (pos) {
      case 'top-left':
        return { top: '1rem', left: '1rem' };
      case 'top-right':
        return { top: '1rem', right: '1rem' };
      case 'bottom-left':
        return { bottom: `${spacing}px`, left: '1rem' };
      case 'bottom-right':
        return { bottom: `${spacing}px`, right: '1rem' };
      default:
        return { bottom: '1rem', right: '1rem' };
    }
  };

  return (
    <button
      className={clsx(
        'fixed w-14 h-14 rounded-full flex items-center justify-center z-40 transition-all duration-300',
        'bg-gradient-to-br from-rose-400 via-rose-500 to-pink-500',
        'text-white text-2xl',
        'shadow-rose-soft hover:shadow-rose-soft-lg',
        'hover:scale-110 hover:-translate-y-1',
        'active:scale-100 active:translate-y-0',
        'group relative'
      )}
      style={getPositionStyle(position, index)}
      onClick={onClick}
      aria-label={ariaLabel}
      title={ariaLabel}
      type="button"
    >
      {/* 发光效果 */}
      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-rose-400 to-pink-500 opacity-0 group-hover:opacity-30 blur-md transition-opacity duration-300"></div>

      {/* 图标 */}
      <span className="relative z-10 group-hover:rotate-12 transition-transform duration-300">
        {icon}
      </span>
    </button>
  );
};
