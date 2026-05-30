import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Pin the workspace root to this folder; there are other lockfiles higher up
  // the tree (the book project, the user's home dir) that Next would otherwise
  // infer as the root.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
