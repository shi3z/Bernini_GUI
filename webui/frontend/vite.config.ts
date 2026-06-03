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
  },
})
