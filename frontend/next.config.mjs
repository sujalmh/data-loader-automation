/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    allowedDevOrigins: ['http://34.41.241.77:8071', 'http://localhost:8071', 'http://127.0.0.1:8071']
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig