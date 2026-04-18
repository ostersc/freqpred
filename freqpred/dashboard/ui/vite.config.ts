import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiPort = env.FREQPRED_API_PORT || '8000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': `http://localhost:${apiPort}`,
      },
    },
  }
})
