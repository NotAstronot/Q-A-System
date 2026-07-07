import useStore from '../store'

export default function LanguageToggle() {
  const { lang, setLang } = useStore()

  return (
    <button
      onClick={() => setLang(lang === 'id' ? 'en' : 'id')}
      className="flex items-center gap-1 text-[10px] font-medium text-gray-400 hover:text-white transition-colors"
      title={lang === 'id' ? 'Switch to English' : 'Ganti ke Indonesia'}
    >
      <span className={lang === 'id' ? 'text-white' : ''}>ID</span>
      <span className="text-gray-600">/</span>
      <span className={lang === 'en' ? 'text-white' : ''}>EN</span>
    </button>
  )
}
