import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // VERY IMPORTANT
  base: './',

  plugins: [react()],

  server: {
    // MUST use 0.0.0.0
    host: '0.0.0.0',

    port: 5173,

    strictPort: true,

    // IMPORTANT for SageMaker/Jupyter proxy
    hmr: false,

    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },

      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})