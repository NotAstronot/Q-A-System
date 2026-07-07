import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { initApiKey } from './api.js'
import './styles/globals.css'

initApiKey()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
