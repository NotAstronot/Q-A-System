import useStore from '../store'

const titles = {
  chat: 'Tanya Jawab',
  upload: 'Upload Dokumen',
  stats: 'Statistik',
}

export default function Header() {
  const { activePage } = useStore()

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <h1 className="text-xl font-semibold text-gray-800">
        {titles[activePage]}
      </h1>
    </header>
  )
}
