/** Chat history panel component */

import React from 'react';
import { Modal } from '../ui/Modal';
import type { DisplayMessage } from '../../types/chat';

interface ChatHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  messages: DisplayMessage[];
  onClear: () => void;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({
  isOpen,
  onClose,
  messages,
  onClear,
}) => {
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('zh-CN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="对话历史">
      <div className="space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            <p>还没有对话记录呢～</p>
          </div>
        ) : (
          <>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`p-3 rounded-lg ${
                    message.isUser
                      ? 'bg-blue-50 border-l-4 border-blue-300'
                      : 'bg-pink-50 border-l-4 border-pink-300'
                  }`}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-xs font-semibold text-gray-600">
                      {message.name || (message.isUser ? '你' : '妹妹')}
                    </span>
                    <span className="text-xs text-gray-400">
                      {formatDate(message.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>
              ))}
            </div>

            <div className="pt-4 border-t border-pink-100">
              <button
                onClick={onClear}
                className="w-full py-2 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors text-sm font-medium"
              >
                清空历史记录
              </button>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
