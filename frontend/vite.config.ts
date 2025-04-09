import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    hmr: {
      clientPort: 443,
      host: "2a8d36cb-3602-46e2-82dc-3d70be763e17-00-1bnxhzpxbgva9.janeway.replit.dev",
    },
  },
});
