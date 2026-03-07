/** Character management page for listing, editing, and deleting characters */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import {
  listUserCharacters,
  deleteUserCharacter,
  updateCharacterPrompt,
} from '../services/characterService';
import { CreateCharacterModal } from '../components/character/CreateCharacterModal';
import type { UserCharacter } from '../types/character';

export const CharacterManagementPage: React.FC = () => {
  const navigate = useNavigate();
  const [characters, setCharacters] = useState<UserCharacter[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingCharacter, setEditingCharacter] = useState<UserCharacter | null>(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);

  useEffect(() => {
    loadCharacters();
  }, []);

  const loadCharacters = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listUserCharacters();
      setCharacters(response.characters);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载角色列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSuccess = (character: UserCharacter) => {
    setCharacters((prev) => [...prev, character]);
    setShowCreateModal(false);
    // Auto-select the new character and navigate to chat
    localStorage.setItem('selectedCharacterId', character.character_id);
    navigate('/');
  };

  const handleDelete = async (characterId: string) => {
    if (!confirm('确定要删除这个角色吗？相关的聊天记录和日记也会被删除。')) {
      return;
    }

    setDeletingId(characterId);
    try {
      await deleteUserCharacter(characterId);
      setCharacters((prev) => prev.filter((c) => c.character_id !== characterId));
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除角色失败');
    } finally {
      setDeletingId(null);
    }
  };

  const handleEditPrompt = async (character: UserCharacter) => {
    setEditingCharacter(character);
    setEditPrompt('');
    setShowEditModal(true);
  };

  const handleSavePrompt = async () => {
    if (!editingCharacter) return;

    setSavingId(editingCharacter.character_id);
    try {
      await updateCharacterPrompt(editingCharacter.character_id, { prompt: editPrompt });
      setShowEditModal(false);
      setEditingCharacter(null);
      setEditPrompt('');
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新提示词失败');
    } finally {
      setSavingId(null);
    }
  };

  const formatDate = (isoDate: string) => {
    return new Date(isoDate).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50/50 via-white to-pink-50/30 dark:from-night-primary dark:via-night-secondary dark:to-night-primary">
      {/* Header - 玻璃态效果 */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-night-secondary/80 backdrop-blur-md border-b border-rose-100/50 dark:border-neutral-800/50 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-neutral-800 dark:text-neutral-100">角色管理</h1>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700 hover:shadow-rose-soft dark:hover:shadow-dark-glow active:scale-95"
            >
              返回聊天
            </button>
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-2.5 rounded-full text-sm font-semibold bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:-translate-y-0.5 transition-all duration-200 active:scale-95"
            >
              创建角色
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-950/50 border-2 border-red-200 dark:border-red-800/50 rounded-xl text-red-700 dark:text-red-200 animate-fade-in">
            <div className="flex items-start justify-between">
              <span>{error}</span>
              <button
                type="button"
                onClick={() => setError(null)}
                className="ml-4 text-sm underline hover:no-underline opacity-70 hover:opacity-100"
              >
                关闭
              </button>
            </div>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-10 h-10 border-4 border-rose-200 dark:border-rose-900/50 border-t-rose-500 dark:border-t-rose-400 rounded-full animate-spin" />
          </div>
        ) : characters.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-6xl mb-4 animate-bounce-soft">🎭</div>
            <h2 className="text-xl font-semibold text-neutral-700 dark:text-neutral-200 mb-2">还没有角色</h2>
            <p className="text-neutral-500 dark:text-neutral-400 mb-6">创建你的第一个角色开始聊天吧</p>
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-3 rounded-full text-sm font-semibold bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:-translate-y-0.5 transition-all duration-200 active:scale-95"
            >
              创建第一个角色
            </button>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {characters.map((character) => (
              <div
                key={character.character_id}
                className="group relative bg-white dark:bg-night-elevated rounded-2xl shadow-soft dark:shadow-dark-soft border border-neutral-100 dark:border-neutral-800 overflow-hidden hover:shadow-soft-lg dark:hover:shadow-dark-soft-lg hover:-translate-y-1 transition-all duration-300"
              >
                {/* 卡片光晕效果 */}
                <div className="absolute inset-0 bg-gradient-to-br from-rose-400/0 to-pink-500/0 group-hover:from-rose-400/5 group-hover:to-pink-500/5 transition-all duration-300 pointer-events-none" />

                <div className="p-6 relative">
                  <h3 className="text-xl font-bold text-neutral-800 dark:text-neutral-100 mb-2">{character.name}</h3>
                  <p className="text-sm text-neutral-500 dark:text-neutral-400 mb-4 font-mono">ID: {character.character_id.slice(0, 8)}...</p>
                  <p className="text-xs text-neutral-400 dark:text-neutral-500 mb-6">
                    创建于 {formatDate(character.created_at)}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        localStorage.setItem('selectedCharacterId', character.character_id);
                        navigate('/');
                      }}
                      className="flex-1 px-4 py-2 rounded-full text-sm font-semibold bg-gradient-to-r from-rose-400 to-rose-500 text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:scale-105 transition-all duration-200"
                    >
                      开始聊天
                    </button>
                    <button
                      type="button"
                      onClick={() => handleEditPrompt(character)}
                      className="px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700 hover:shadow-rose-soft dark:hover:shadow-dark-glow active:scale-95 disabled:opacity-50"
                      disabled={savingId === character.character_id}
                    >
                      {savingId === character.character_id ? '保存中...' : '编辑'}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(character.character_id)}
                      className="px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-red-200 dark:border-red-800/50 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/50 hover:shadow-red-200 dark:hover:shadow-red-900/50 active:scale-95 disabled:opacity-50"
                      disabled={deletingId === character.character_id}
                    >
                      {deletingId === character.character_id ? '删除中...' : '删除'}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Create Character Modal */}
      <CreateCharacterModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={handleCreateSuccess}
      />

      {/* Edit Prompt Modal - 精美的玻璃态效果 */}
      {showEditModal && editingCharacter && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 dark:bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="bg-white/95 dark:bg-night-elevated/95 backdrop-blur-md rounded-2xl shadow-2xl dark:shadow-dark-soft-lg w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col border border-neutral-100 dark:border-neutral-700 animate-scale-in">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-rose-100 dark:border-neutral-700">
              <h2 className="text-xl font-bold text-neutral-800 dark:text-neutral-100">
                编辑 <span className="text-gradient">{editingCharacter.name}</span> 的提示词
              </h2>
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingCharacter(null);
                  setEditPrompt('');
                }}
                className="p-2 rounded-full hover:bg-rose-50 dark:hover:bg-rose-950/50 transition-colors group"
              >
                <svg className="w-5 h-5 text-neutral-500 dark:text-neutral-400 group-hover:text-rose-500 dark:group-hover:text-rose-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6">
              <label htmlFor="edit-prompt" className="block text-sm font-semibold text-neutral-700 dark:text-neutral-200 mb-2">
                角色提示词
              </label>
              <textarea
                id="edit-prompt"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                placeholder="输入新的角色提示词..."
                rows={15}
                className="w-full px-4 py-3 bg-white dark:bg-night-secondary border-2 border-neutral-200 dark:border-neutral-600 rounded-xl focus:ring-2 focus:ring-rose-400 dark:focus:ring-rose-600 focus:border-rose-400 dark:focus:border-rose-600 resize-none text-neutral-800 dark:text-neutral-100 placeholder:text-neutral-400 dark:placeholder:text-neutral-500 transition-all duration-200"
              />
            </div>

            {/* Modal Footer */}
            <div className="flex justify-end gap-3 p-6 border-t border-rose-100 dark:border-neutral-700 bg-neutral-50/50 dark:bg-neutral-900/30">
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingCharacter(null);
                  setEditPrompt('');
                }}
                className="px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 bg-white/80 dark:bg-neutral-800/80 backdrop-blur-sm border-2 border-neutral-200 dark:border-neutral-600 text-neutral-700 dark:text-neutral-200 hover:border-rose-300 dark:hover:border-rose-700 hover:shadow-rose-soft dark:hover:shadow-dark-glow active:scale-95"
              >
                取消
              </button>
              <button
                type="button"
                onClick={handleSavePrompt}
                disabled={!editPrompt.trim() || savingId !== null}
                className={clsx(
                  "px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 active:scale-95",
                  editPrompt.trim() && savingId === null
                    ? "bg-gradient-to-r from-rose-400 via-rose-500 to-pink-500 text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:-translate-y-0.5"
                    : "bg-neutral-200 dark:bg-neutral-700 text-neutral-400 dark:text-neutral-500 cursor-not-allowed"
                )}
              >
                {savingId ? '保存中...' : '保存'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
