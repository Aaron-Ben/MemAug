/** Character selector component for switching between characters */

import React, { useState, useEffect } from 'react';
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
    <div className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold cursor-pointer transition-all duration-300 border-2 shadow-lg hover:-translate-y-0.5 hover:shadow-xl active:translate-y-0 bg-gradient-to-r from-pink-300 to-pink-500 border-pink-200 text-white shadow-pink-300/30 hover:from-pink-500 hover:to-pink-600 hover:shadow-pink-400/40"
        aria-label="Select character"
        aria-expanded={isOpen}
      >
        <span className="text-[13px] font-semibold">
          {loading ? '加载中...' : selectedCharacter?.name || '选择角色'}
        </span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
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
          <div className="absolute left-0 mt-2 w-64 bg-white rounded-lg shadow-xl border border-pink-200 z-20">
            <div className="p-2 max-h-96 overflow-y-auto">
              {characters.length === 0 ? (
                <div className="p-4 text-center text-gray-500 text-sm">
                  暂无角色
                </div>
              ) : (
                characters.map((character) => (
                  <button
                    key={character.character_id}
                    type="button"
                    onClick={() => handleSelect(character)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                      selectedCharacterId === character.character_id
                        ? 'bg-pink-100 text-pink-700'
                        : 'hover:bg-pink-50 text-gray-700'
                    }`}
                  >
                    <div className="font-medium">{character.name}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {character.character_id.slice(0, 8)}...
                    </div>
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
