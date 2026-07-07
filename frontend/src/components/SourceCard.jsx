import useStore from '../store'
import { FileText } from 'lucide-react'

export default function SourceCard({ source }) {
  const { tr } = useStore()

  return (
    <div className="inline-flex items-center gap-1.5 px-2 py-1 bg-gray-100 border border-gray-200 rounded-md text-xs text-gray-600 hover:bg-gray-200 transition-colors">
      <FileText size={12} />
      <span className="font-medium">{source.filename}</span>
      <span className="text-gray-400">{tr('source.page')} {source.page}</span>
    </div>
  )
}
