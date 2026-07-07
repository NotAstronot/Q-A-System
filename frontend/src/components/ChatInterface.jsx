import { useState, useRef, useEffect } from 'react'
import useStore from '../store'
import MessageBubble from './MessageBubble'
import { Send, Loader2 } from 'lucide-react'

export default function ChatInterface() {
  const [input, setInput] = useState('')
  const { messages, isLoading, sendMessage, clearMessages, tr } = useStore()
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || isLoading) return
    sendMessage(trimmed)
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-4">
              <Send className="text-blue-500" size={28} />
            </div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">{tr('chat.empty_title')}</h3>
            <p className="text-sm text-gray-500 max-w-md">
              {tr('chat.empty_desc')}
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm pl-2">
            <Loader2 size={16} className="animate-spin" />
            <span>{tr('chat.processing')}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-gray-200 bg-white p-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={tr('chat.placeholder')}
            disabled={isLoading}
            className="flex-1 px-4 py-3 border border-gray-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-5 py-3 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors disabled:bg-blue-300 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
            {tr('chat.send')}
          </button>
        </form>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="mt-2 text-xs text-gray-400 hover:text-gray-600"
          >
            {tr('chat.clear')}
          </button>
        )}
      </div>
    </div>
  )
}
