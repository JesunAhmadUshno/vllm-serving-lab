import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/ result files at build time, so every number on the page is generated, never typed.
export default defineConfig({
  base: '/vllm-serving-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
