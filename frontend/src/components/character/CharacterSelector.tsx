/** Character selector component for switching between characters */

import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import { listAllCharacters } from '../../services/characterService';
import type { UserCharacter } from '../../types/character';

interface CharacterSelectorProps {
  selectedCharacterId: string;
  onCharacterChange: (characterId: string) => void;
  className?: string;
}

export const CharacterSelector: React.FC<CharacterSelectorProps> = ({
  selectedCharacterId,
  onCharacterChange,
  className = '',
}) => {
  const [characters, setCharacters] = useState<UserCharacter[]>([]);
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Load characters when opening the dropdown
    if (isOpen) {
      loadCharacters();
    }
  }, [isOpen]);

  const loadCharacters = async () => {
    try {
      const chars = await listAllCharacters();
      setCharacters(chars);
    } catch (error) {
      console.error('Failed to load characters:', error);
    } finally {
      setLoading(false);
    }
  };

  const selectedCharacter = characters.find(c => c.character_id === selectedCharacterId);

  const handleSelect = (character: UserCharacter) => {
    onCharacterChange(character.character_id);
    setIsOpen(false);
  };

  return (
    <div className={clsx('relative', className)}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold cursor-pointer transition-all duration-200 border-2 shadow-sm hover:shadow-md active:scale-95",
          selectedCharacterId
            ? "bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 border-rose-300 text-white shadow-rose-soft hover:shadow-rose-soft-lg"
            : "bg-white/90 dark:bg-neutral-800/90 backdrop-blur-sm border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700"
        )}
        aria-label="选择角色"
        aria-expanded={isOpen}
      >
        <span className="text-[13px] font-semibold">
          {loading ? '...' : selectedCharacter?.name || '选择角色'}
        </span>
        <svg
          className={clsx('w-3.5 h-3.5 transition-transform duration-200', isOpen && 'rotate-180')}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute left-0 mt-2 w-56 bg-white/95 dark:bg-night-elevated/95 backdrop-blur-md rounded-xl shadow-xl dark:shadow-dark-soft-lg border border-neutral-200 dark:border-neutral-700 z-20 animate-fade-in">
            <div className="p-2 max-h-80 overflow-y-auto scrollbar-elegant">
              {characters.length === 0 ? (
                <div className="p-4 text-center text-neutral-500 dark:text-neutral-400 text-sm">
                  暂无角色
                </div>
              ) : (
                characters.map((character) => (
                  <button
                    key={character.character_id}
                    type="button"
                    onClick={() => handleSelect(character)}
                    className={clsx(
                      "w-full text-left px-3 py-2.5 rounded-lg transition-all duration-200",
                      selectedCharacterId === character.character_id
                        ? "bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/50 dark:to-pink-900/50 text-rose-700 dark:text-rose-200 font-semibold"
                        : "text-neutral-700 dark:text-neutral-200 hover:bg-rose-50 dark:hover:bg-rose-950/30"
                    )}
                  >
                    <div className="font-medium text-sm">{character.name}</div>
                  </button>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
