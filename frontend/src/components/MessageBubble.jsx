import ReactMarkdown from 'react-markdown'
import useStore from '../store'
import SourceCard from './SourceCard'
import { User, Bot } from 'lucide-react'

function isSafeUrl(url) {
  try {
    const parsed = new URL(url, window.location.origin)
    return ['http:', 'https:', 'mailto:'].includes(parsed.protocol)
  } catch {
    return false
  }
}

function LinkRenderer({ href, children }) {
  if (!href || !isSafeUrl(href)) {
    return <span className="text-gray-600">{children}</span>
  }
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-600 underline hover:text-blue-800"
    >
      {children}
    </a>
  )
}

function ImageRenderer({ src, alt }) {
  if (!src || !isSafeUrl(src)) {
    return null
  }
  return <img src={src} alt={alt || ''} className="max-w-full rounded" loading="lazy" />
}

export default function MessageBubble({ message }) {
  const { tr } = useStore()
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
              <ReactMarkdown
                components={{
                  a: LinkRenderer,
                  img: ImageRenderer,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-400 px-1">{tr('chat.sources')}</p>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((source, i) => (
                <SourceCard key={i} source={source} />
              ))}
            </div>
          </div>
        )}

        {!isUser && message.rewritten_query && message.rewritten_query !== message.content && (
          <div className="mt-2 px-1">
            <p className="text-xs text-gray-400 italic">
              {tr('chat.rewritten')} &ldquo;{message.rewritten_query}&rdquo;
            </p>
          </div>
        )}

        {!isUser && message.attempts > 1 && (
          <p className="text-xs text-gray-400 mt-1 px-1">
            {tr('chat.attempts', { n: message.attempts })}
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
