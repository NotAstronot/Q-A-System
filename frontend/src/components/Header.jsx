import useStore from '../store'

export default function Header() {
  const { activePage, tr, features } = useStore()

  const pageTitle = {
    chat: tr('header.chat'),
    upload: tr('header.upload'),
    stats: tr('header.stats'),
  }

  const featureLabels = []
  if (features?.hybrid_search) featureLabels.push('Hybrid')
  if (features?.reranking) featureLabels.push('Rerank')
  if (features?.query_rewriting) featureLabels.push('Rewrite')
  if (features?.table_parsing) featureLabels.push('Table')
  if (features?.parent_child_chunking) featureLabels.push('P-C Chunk')

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <h1 className="text-xl font-semibold text-gray-800">
        {pageTitle[activePage]}
      </h1>
      <div className="flex items-center gap-2">
        {features && (
          <>
            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
              features.provider === 'ollama'
                ? 'bg-green-100 text-green-700'
                : features.provider === '9router'
                ? 'bg-purple-100 text-purple-700'
                : 'bg-blue-100 text-blue-700'
            }`}>
              {features.provider === 'ollama'
                ? 'Ollama'
                : features.provider === '9router'
                ? '9Router'
                : 'OpenRouter'}
            </span>
            <span className="text-xs text-gray-400 font-mono">
              {features.model}
            </span>
            <div className="flex gap-1 ml-2">
              {featureLabels.map((label) => (
                <span key={label} className="text-[10px] bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
                  {label}
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </header>
  )
}
