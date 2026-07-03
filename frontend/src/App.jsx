import { useEffect } from 'react'
import useStore from './store'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ChatInterface from './components/ChatInterface'
import UploadForm from './components/UploadForm'
import StatsPanel from './components/StatsPanel'

export default function App() {
  const { activePage, fetchDocuments, fetchStats, error, clearError } = useStore()

  useEffect(() => {
    fetchDocuments()
    fetchStats()
  }, [])

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        {error && (
          <div className="mx-4 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
            <span className="text-red-700 text-sm">{error}</span>
            <button onClick={clearError} className="text-red-500 hover:text-red-700 text-sm font-medium">
              Tutup
            </button>
          </div>
        )}
        <main className="flex-1 overflow-hidden">
          {activePage === 'chat' && <ChatInterface />}
          {activePage === 'upload' && <UploadForm />}
          {activePage === 'stats' && <StatsPanel />}
        </main>
      </div>
    </div>
  )
}
