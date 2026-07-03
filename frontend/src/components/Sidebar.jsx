import useStore from '../store'
import { MessageSquare, Upload, BarChart3, FileText } from 'lucide-react'

const navItems = [
  { id: 'chat', label: 'Tanya Jawab', icon: MessageSquare },
  { id: 'upload', label: 'Upload Dokumen', icon: Upload },
  { id: 'stats', label: 'Statistik', icon: BarChart3 },
]

export default function Sidebar() {
  const { activePage, setActivePage, documents } = useStore()

  return (
    <aside className="w-64 bg-gray-900 text-white flex flex-col">
      <div className="p-5 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center font-bold text-lg">
            QA
          </div>
          <div>
            <h2 className="font-semibold text-sm">Internal QA</h2>
            <p className="text-xs text-gray-400">System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActivePage(id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
              activePage === id
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <Icon size={18} />
            {label}
          </button>
        ))}
      </nav>

      <div className="p-3 border-t border-gray-700">
        <p className="text-xs text-gray-500 mb-2 px-2">Dokumen ({documents.length})</p>
        <div className="max-h-40 overflow-y-auto scrollbar-thin space-y-1">
          {documents.map((doc) => (
            <div
              key={doc.filename}
              className="flex items-center gap-2 px-2 py-1.5 rounded text-xs text-gray-400 hover:bg-gray-800"
            >
              <FileText size={14} />
              <span className="truncate">{doc.filename}</span>
            </div>
          ))}
          {documents.length === 0 && (
            <p className="text-xs text-gray-600 px-2">Belum ada dokumen</p>
          )}
        </div>
      </div>
    </aside>
  )
}
