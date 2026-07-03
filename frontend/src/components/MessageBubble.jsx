import ReactMarkdown from 'react-markdown'
import SourceCard from './SourceCard'
import { User, Bot } from 'lucide-react'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
          <Bot size={16} className="text-white" />
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? 'order-1' : ''}`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'
          }`}
        >
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <div className="prose prose-sm prose-gray max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-li:my-0">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-400 px-1">Sumber:</p>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((source, i) => (
                <SourceCard key={i} source={source} />
              ))}
            </div>
          </div>
        )}

        {!isUser && message.attempts > 1 && (
          <p className="text-xs text-gray-400 mt-1 px-1">
            Dibuat dalam {message.attempts} percobaan (sitasi diperkuat)
          </p>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
          <User size={16} className="text-gray-600" />
        </div>
      )}
    </div>
  )
}
