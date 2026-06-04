import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    // Accessed over the tailnet under various hostnames; allow any host so
    // vite's host check doesn't block tailnet names (e.g. "tsuginosuke").
    allowedHosts: true,
    // Proxy API/WS to the backend so the browser talks to a single origin
    // (the page's own scheme/host). Avoids CORS and HTTPS-page -> HTTP-API
    // mixed-content blocks (Safari "Load failed") when served via Tailscale.
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'http://localhost:8000', ws: true, changeOrigin: true },
    },
  },
})
