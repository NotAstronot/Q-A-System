const id = {
  // Sidebar
  'nav.chat': 'Tanya Jawab',
  'nav.upload': 'Upload Dokumen',
  'nav.stats': 'Statistik',
  'sidebar.documents': 'Dokumen',
  'sidebar.no_docs': 'Belum ada dokumen',
  'sidebar.title': 'Internal QA',
  'sidebar.subtitle': 'System',

  // Header
  'header.chat': 'Tanya Jawab',
  'header.upload': 'Upload Dokumen',
  'header.stats': 'Statistik',

  // Chat
  'chat.placeholder': 'Ketik pertanyaan Anda...',
  'chat.send': 'Kirim',
  'chat.processing': 'Memproses jawaban...',
  'chat.clear': 'Hapus percakapan',
  'chat.empty_title': 'Mulai Bertanya',
  'chat.empty_desc': 'Ajukan pertanyaan tentang dokumen internal perusahaan. Sistem akan menjawab berdasarkan dokumen yang tersedia.',
  'chat.sources': 'Sumber:',
  'chat.rewritten': 'Query diperluas:',
  'chat.attempts': 'Dibuat dalam {n} percobaan (sitasi diperkuat)',
  'chat.error_prefix': 'Maaf, terjadi kesalahan:',

  // Upload
  'upload.dragdrop': 'Seret & lepas file PDF ke sini',
  'upload.or': 'atau',
  'upload.select': 'Pilih File PDF',
  'upload.max_size': 'Maksimal ukuran file: 10MB',
  'upload.uploading': 'Mengunggah dokumen...',
  'upload.success_title': 'Upload berhasil!',
  'upload.success_detail': '{name} — {n} chunks dibuat',
  'upload.fail_title': 'Upload gagal',
  'upload.ingested_title': 'Dokumen yang sudah di-ingest',

  // Stats
  'stats.title': 'Statistik Sistem',
  'stats.collection': 'Collection',
  'stats.total_chunks': 'Total Chunks',
  'stats.documents': 'Dokumen',
  'stats.provider': 'Provider',
  'stats.detail_title': 'Detail',
  'stats.collection_name': 'Collection Name',
  'stats.documents_dir': 'Documents Directory',
  'stats.total_files': 'Total Files',
  'stats.bm25': 'BM25 Index',
  'stats.bm25_trained': 'Terlatih',
  'stats.bm25_untrained': 'Belum',
  'stats.features_title': 'Fitur Aktif',

  // Source card
  'source.page': 'Hal.',

  // Error banner
  'error.close': 'Tutup',

  // LanguageToggle
  'lang.en': 'EN',
  'lang.id': 'ID',
}

const en = {
  'nav.chat': 'Q&A',
  'nav.upload': 'Upload Document',
  'nav.stats': 'Statistics',
  'sidebar.documents': 'Documents',
  'sidebar.no_docs': 'No documents yet',
  'sidebar.title': 'Internal QA',
  'sidebar.subtitle': 'System',

  'header.chat': 'Q&A',
  'header.upload': 'Upload Document',
  'header.stats': 'Statistics',

  'chat.placeholder': 'Type your question...',
  'chat.send': 'Send',
  'chat.processing': 'Processing answer...',
  'chat.clear': 'Clear conversation',
  'chat.empty_title': 'Start Asking',
  'chat.empty_desc': 'Ask questions about internal company documents. The system will answer based on available documents.',
  'chat.sources': 'Sources:',
  'chat.rewritten': 'Query expanded:',
  'chat.attempts': 'Generated in {n} attempt(s) (citation enforced)',
  'chat.error_prefix': 'Sorry, an error occurred:',

  'upload.dragdrop': 'Drag & drop PDF files here',
  'upload.or': 'or',
  'upload.select': 'Select PDF File',
  'upload.max_size': 'Maximum file size: 10MB',
  'upload.uploading': 'Uploading document...',
  'upload.success_title': 'Upload successful!',
  'upload.success_detail': '{name} — {n} chunks created',
  'upload.fail_title': 'Upload failed',
  'upload.ingested_title': 'Ingested documents',

  'stats.title': 'System Statistics',
  'stats.collection': 'Collection',
  'stats.total_chunks': 'Total Chunks',
  'stats.documents': 'Documents',
  'stats.provider': 'Provider',
  'stats.detail_title': 'Details',
  'stats.collection_name': 'Collection Name',
  'stats.documents_dir': 'Documents Directory',
  'stats.total_files': 'Total Files',
  'stats.bm25': 'BM25 Index',
  'stats.bm25_trained': 'Trained',
  'stats.bm25_untrained': 'Not yet',
  'stats.features_title': 'Active Features',

  'source.page': 'Page',

  'error.close': 'Close',

  'lang.en': 'EN',
  'lang.id': 'ID',
}

const translations = { id, en }

export function t(key, lang, params = {}) {
  let text = translations[lang]?.[key] ?? translations['id'][key] ?? key
  for (const [k, v] of Object.entries(params)) {
    text = text.replace(`{${k}}`, v)
  }
  return text
}

export const LANGS = [
  { code: 'id', label: 'ID' },
  { code: 'en', label: 'EN' },
]
