import { useState, useCallback } from 'react'
import useStore from '../store'
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'

export default function UploadForm() {
  const [dragActive, setDragActive] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const { uploadFile, isUploading, documents, tr } = useStore()

  const handleUpload = async (file) => {
    setUploadResult(null)
    try {
      const result = await uploadFile(file)
      setUploadResult({ success: true, ...result })
    } catch (err) {
      setUploadResult({ success: false, error: err.message })
    }
  }

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(async (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const file = e.dataTransfer.files[0]
    if (file && file.type === 'application/pdf') {
      await handleUpload(file)
    }
  }, [uploadFile])

  const handleFileSelect = async (e) => {
    const file = e.target.files[0]
    if (file) {
      await handleUpload(file)
    }
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors ${
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        {isUploading ? (
          <div className="flex flex-col items-center">
            <Loader2 size={48} className="text-blue-500 animate-spin mb-4" />
            <p className="text-gray-600 font-medium">{tr('upload.uploading')}</p>
          </div>
        ) : (
          <>
            <Upload size={48} className="mx-auto text-gray-400 mb-4" />
            <p className="text-gray-700 font-medium mb-2">
              {tr('upload.dragdrop')}
            </p>
            <p className="text-sm text-gray-500 mb-4">{tr('upload.or')}</p>
            <label className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors cursor-pointer">
              <FileText size={16} />
              {tr('upload.select')}
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
            <p className="text-xs text-gray-400 mt-4">{tr('upload.max_size')}</p>
          </>
        )}
      </div>

      {uploadResult && (
        <div className={`mt-4 p-4 rounded-xl flex items-start gap-3 ${
          uploadResult.success
            ? 'bg-green-50 border border-green-200'
            : 'bg-red-50 border border-red-200'
        }`}>
          {uploadResult.success ? (
            <>
              <CheckCircle size={20} className="text-green-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-green-700 font-medium text-sm">{tr('upload.success_title')}</p>
                <p className="text-green-600 text-xs mt-1">
                  {tr('upload.success_detail', { name: uploadResult.filename, n: uploadResult.chunks_created })}
                </p>
              </div>
            </>
          ) : (
            <>
              <AlertCircle size={20} className="text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-700 font-medium text-sm">{tr('upload.fail_title')}</p>
                <p className="text-red-600 text-xs mt-1">{uploadResult.error}</p>
              </div>
            </>
          )}
        </div>
      )}

      {documents.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            {tr('upload.ingested_title')} ({documents.length})
          </h3>
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.filename}
                className="flex items-center justify-between px-4 py-3 bg-white border border-gray-200 rounded-xl"
              >
                <div className="flex items-center gap-3">
                  <FileText size={18} className="text-gray-400" />
                  <span className="text-sm text-gray-700">{doc.filename}</span>
                </div>
                <span className="text-xs text-gray-400">{doc.size_kb} KB</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
