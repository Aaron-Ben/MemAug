/** Character management page for listing, editing, and deleting characters */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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
    <div className="min-h-screen bg-gradient-to-br from-pink-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800">角色管理</h1>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-full text-sm font-semibold border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
            >
              返回聊天
            </button>
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-2.5 rounded-full text-sm font-semibold bg-gradient-to-r from-pink-400 to-pink-600 text-white hover:from-pink-500 hover:to-pink-700 shadow-lg shadow-pink-300/30 transition-all"
            >
              创建角色
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
            <button
              type="button"
              onClick={() => setError(null)}
              className="ml-4 underline"
            >
              关闭
            </button>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-8 h-8 border-4 border-pink-200 border-t-pink-600 rounded-full animate-spin" />
          </div>
        ) : characters.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">🎭</div>
            <h2 className="text-xl font-semibold text-gray-700 mb-2">还没有角色</h2>
            <p className="text-gray-500 mb-6">创建你的第一个角色开始聊天吧</p>
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-3 rounded-full text-sm font-semibold bg-gradient-to-r from-pink-400 to-pink-600 text-white hover:from-pink-500 hover:to-pink-700 shadow-lg shadow-pink-300/30 transition-all"
            >
              创建第一个角色
            </button>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {characters.map((character) => (
              <div
                key={character.character_id}
                className="bg-white rounded-2xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
              >
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-800 mb-2">{character.name}</h3>
                  <p className="text-sm text-gray-500 mb-4">ID: {character.character_id.slice(0, 8)}...</p>
                  <p className="text-xs text-gray-400 mb-6">
                    创建于 {formatDate(character.created_at)}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        localStorage.setItem('selectedCharacterId', character.character_id);
                        navigate('/');
                      }}
                      className="flex-1 px-4 py-2 rounded-full text-sm font-semibold bg-gradient-to-r from-pink-400 to-pink-600 text-white hover:from-pink-500 hover:to-pink-700 transition-all"
                    >
                      开始聊天
                    </button>
                    <button
                      type="button"
                      onClick={() => handleEditPrompt(character)}
                      className="px-4 py-2 rounded-full text-sm font-semibold border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
                      disabled={savingId === character.character_id}
                    >
                      {savingId === character.character_id ? '保存中...' : '编辑'}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(character.character_id)}
                      className="px-4 py-2 rounded-full text-sm font-semibold border-2 border-red-300 text-red-600 hover:bg-red-50 transition-colors disabled:opacity-50"
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

      {/* Edit Prompt Modal */}
      {showEditModal && editingCharacter && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-6 border-b border-pink-100">
              <h2 className="text-xl font-bold text-gray-800">
                编辑 {editingCharacter.name} 的提示词
              </h2>
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingCharacter(null);
                  setEditPrompt('');
                }}
                className="p-2 rounded-full hover:bg-pink-100 transition-colors"
              >
                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              <label htmlFor="edit-prompt" className="block text-sm font-medium text-gray-700 mb-2">
                角色提示词
              </label>
              <textarea
                id="edit-prompt"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                placeholder="输入新的角色提示词..."
                rows={15}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-pink-500 resize-none"
              />
            </div>

            <div className="flex justify-end gap-3 p-6 border-t border-pink-100">
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingCharacter(null);
                  setEditPrompt('');
                }}
                className="px-6 py-2.5 rounded-full text-sm font-semibold border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
              >
                取消
              </button>
              <button
                type="button"
                onClick={handleSavePrompt}
                disabled={!editPrompt.trim() || savingId !== null}
                className="px-6 py-2.5 rounded-full text-sm font-semibold bg-gradient-to-r from-pink-400 to-pink-600 text-white hover:from-pink-500 hover:to-pink-700 shadow-lg shadow-pink-300/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
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
