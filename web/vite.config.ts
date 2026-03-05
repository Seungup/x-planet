import { defineConfig } from "vite";

export default defineConfig({
  base: "/x-planet/",
  server: {
    port: 8080,
    host: "0.0.0.0",
  },
  build: {
    target: "es2022",
  },
});
