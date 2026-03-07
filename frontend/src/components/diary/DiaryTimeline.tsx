/** Diary timeline component with month navigation and grouping - Refined elegant style */

import React, { useState, useEffect } from 'react';
import { DiaryEntry, DiaryGroup } from '../../services/diaryService';
import { MonthGroup } from './MonthGroup';

interface DiaryTimelineProps {
  diaries: DiaryEntry[];
  onSelectDiary: (diary: DiaryEntry) => void;
  onEditDiary: (diary: DiaryEntry, e: React.MouseEvent) => void;
  onDeleteDiary: (diary: DiaryEntry, e: React.MouseEvent) => void;
}

export const DiaryTimeline: React.FC<DiaryTimelineProps> = ({
  diaries,
  onSelectDiary,
  onEditDiary,
  onDeleteDiary,
}) => {
  const [groups, setGroups] = useState<DiaryGroup[]>([]);
  const [activeMonth, setActiveMonth] = useState<string>('');

  // Initialize groups when diaries change
  useEffect(() => {
    // Import here to avoid circular dependency
    import('../../services/diaryService').then(({ groupDiariesByMonth }) => {
      const grouped = groupDiariesByMonth(diaries);
      setGroups(grouped);
      if (grouped.length > 0) {
        setActiveMonth(`${grouped[0].year}-${grouped[0].month}`);
      }
    });
  }, [diaries]);

  // Toggle expand/collapse for a group
  const toggleGroup = (index: number) => {
    setGroups((prev) =>
      prev.map((group, i) =>
        i === index ? { ...group, expanded: !group.expanded } : group
      )
    );
  };

  // Scroll to specific month
  const scrollToMonth = (year: number, month: number) => {
    const key = `${year}-${month}`;
    setActiveMonth(key);

    // Expand that month
    setGroups((prev) =>
      prev.map((group) =>
        group.year === year && group.month === month
          ? { ...group, expanded: true }
          : group
      )
    );

    // Scroll to element
    const element = document.getElementById(`month-${key}`);
    element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  if (groups.length === 0) {
    return (
      <div className="text-center text-amber-600/70 dark:text-amber-400/60 py-8">
        <p className="text-lg mb-2">还没有日记呢～</p>
        <p className="text-sm">妹妹会记录和哥哥的重要时刻</p>
      </div>
    );
  }

  return (
    <div className="flex gap-5 items-start">
      {/* Side navigation */}
      <div className="sticky top-4 w-32 flex-shrink-0 max-h-[calc(80vh-32px)] overflow-y-auto">
        <div className="bg-amber-50 dark:bg-stone-800 rounded-2xl border border-amber-200/50 dark:border-stone-600 p-2 shadow-sm">
          {groups.map((group) => (
            <div
              key={`${group.year}-${group.month}`}
              className={`
                px-3 py-2 rounded-xl cursor-pointer transition-all duration-200 text-sm mb-1 last:mb-0
                ${activeMonth === `${group.year}-${group.month}`
                  ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white shadow-sm font-medium'
                  : 'text-amber-700 dark:text-amber-400 hover:bg-amber-100/60 dark:hover:bg-amber-950/30'
                }
              `}
              onClick={() => scrollToMonth(group.year, group.month)}
            >
              <div className="text-xs font-medium">
                {group.year}年{group.month}月
              </div>
              <div className={clsx(
                'text-[10px]',
                activeMonth === `${group.year}-${group.month}` ? 'text-white/70' : 'text-amber-500/70'
              )}>
                {group.count}篇
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline content */}
      <div className="flex-1 min-w-0">
        {groups.map((group, index) => (
          <MonthGroup
            key={`${group.year}-${group.month}`}
            group={group}
            onToggle={() => toggleGroup(index)}
            onSelectDiary={onSelectDiary}
            onEditDiary={onEditDiary}
            onDeleteDiary={onDeleteDiary}
          />
        ))}
      </div>
    </div>
  );
};

// Helper for clsx
function clsx(...classes: (string | boolean | undefined | null)[]) {
  return classes.filter(Boolean).join(' ');
}
