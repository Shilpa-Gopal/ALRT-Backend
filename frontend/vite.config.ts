import { defineConfig } from "vite";
import react from "@vitejs/plugin-react"; // If you're using React (adjust if not)

export default defineConfig({
  plugins: [react()], // If you're using React (adjust if not)
  server: {
    host: "0.0.0.0",
    hmr: {
      clientPort: 443,
    },
    allowedHosts: [
      "2a8d36cb-3602-46e2-82dc-3d70be763e17-00-1bnxhzpxbgva9.janeway.replit.dev",
      ".replit.dev", // This will allow all subdomains of replit.dev
    ],
  },
});
