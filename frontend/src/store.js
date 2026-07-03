import { create } from 'zustand'
import { queryDocuments, uploadDocument, getStats, getDocuments } from './api'

const useStore = create((set, get) => ({
  messages: [],
  documents: [],
  stats: null,
  isLoading: false,
  isUploading: false,
  error: null,
  activePage: 'chat',

  setActivePage: (page) => set({ activePage: page }),

  sendMessage: async (question) => {
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: question }
    set((state) => ({
      messages: [...state.messages, userMessage],
      isLoading: true,
      error: null,
    }))

    try {
      const result = await queryDocuments(question)
      const aiMessage = {
        id: crypto.randomUUID(),
        role: 'ai',
        content: result.answer,
        sources: result.sources,
        validation: result.validation,
        attempts: result.attempts,
      }
      set((state) => ({
        messages: [...state.messages, aiMessage],
        isLoading: false,
      }))
    } catch (err) {
      const errorMessage = {
        id: crypto.randomUUID(),
        role: 'ai',
        content: `Maaf, terjadi kesalahan: ${err.message}`,
        sources: [],
        validation: { valid: false, citation_count: 0 },
        attempts: 0,
      }
      set((state) => ({
        messages: [...state.messages, errorMessage],
        isLoading: false,
        error: err.message,
      }))
    }
  },

  uploadFile: async (file) => {
    set({ isUploading: true, error: null })
    try {
      const result = await uploadDocument(file)
      set({ isUploading: false })
      await get().fetchDocuments()
      await get().fetchStats()
      return result
    } catch (err) {
      set({ isUploading: false, error: err.message })
      throw err
    }
  },

  fetchDocuments: async () => {
    try {
      const result = await getDocuments()
      set({ documents: result.documents })
    } catch (err) {
      set({ error: err.message })
    }
  },

  fetchStats: async () => {
    try {
      const result = await getStats()
      set({ stats: result })
    } catch (err) {
      set({ error: err.message })
    }
  },

  clearMessages: () => set({ messages: [] }),

  clearError: () => set({ error: null }),
}))

export default useStore
