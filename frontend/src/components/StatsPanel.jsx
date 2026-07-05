import useStore from '../store'
import { Database, FileText, HardDrive, RefreshCw, CheckCircle, XCircle, Cpu } from 'lucide-react'

export default function StatsPanel() {
  const { stats, documents, features, fetchStats, fetchDocuments } = useStore()

  const handleRefresh = () => {
    fetchStats()
    fetchDocuments()
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-800">Statistik Sistem</h2>
        <button
          onClick={handleRefresh}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Database size={20} className="text-blue-600" />
            </div>
            <span className="text-sm text-gray-500">Collection</span>
          </div>
          <p className="text-2xl font-bold text-gray-800 truncate">
            {stats?.collection_name || '-'}
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <HardDrive size={20} className="text-green-600" />
            </div>
            <span className="text-sm text-gray-500">Total Chunks</span>
          </div>
          <p className="text-2xl font-bold text-gray-800">
            {stats?.total_chunks ?? '-'}
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <FileText size={20} className="text-purple-600" />
            </div>
            <span className="text-sm text-gray-500">Dokumen</span>
          </div>
          <p className="text-2xl font-bold text-gray-800">
            {documents.length}
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
              <Cpu size={20} className="text-amber-600" />
            </div>
            <span className="text-sm text-gray-500">Provider</span>
          </div>
          <p className="text-2xl font-bold text-gray-800">
            {stats?.provider === 'ollama' ? 'Ollama' : 'OpenRouter'}
          </p>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Detail</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between py-2 border-b border-gray-100">
            <span className="text-gray-500">Collection Name</span>
            <span className="text-gray-800 font-medium">{stats?.collection_name || '-'}</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-100">
            <span className="text-gray-500">Total Chunks</span>
            <span className="text-gray-800 font-medium">{stats?.total_chunks ?? '-'}</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-100">
            <span className="text-gray-500">Documents Directory</span>
            <span className="text-gray-800 font-medium">{stats?.documents_dir || '-'}</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-100">
            <span className="text-gray-500">Total Files</span>
            <span className="text-gray-800 font-medium">{documents.length} file(s)</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-100">
            <span className="text-gray-500">BM25 Index</span>
            <span className={`font-medium flex items-center gap-1 ${stats?.bm25_trained ? 'text-green-600' : 'text-red-500'}`}>
              {stats?.bm25_trained ? (
                <><CheckCircle size={14} /> Terlatih</>
              ) : (
                <><XCircle size={14} /> Belum</>
              )}
            </span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-500">Provider</span>
            <span className="text-gray-800 font-medium">{stats?.provider || '-'}</span>
          </div>
        </div>
      </div>

      {features && (
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Fitur Aktif</h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Hybrid Search', active: features.hybrid_search },
              { label: 'Reranking', active: features.reranking },
              { label: 'Query Rewriting', active: features.query_rewriting },
              { label: 'Table Parsing', active: features.table_parsing },
              { label: 'Parent-Child Chunk', active: features.parent_child_chunking },
            ].map(({ label, active }) => (
              <div key={label} className="flex items-center gap-2 text-sm">
                {active ? (
                  <CheckCircle size={16} className="text-green-500" />
                ) : (
                  <XCircle size={16} className="text-gray-300" />
                )}
                <span className={active ? 'text-gray-700' : 'text-gray-400'}>{label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
