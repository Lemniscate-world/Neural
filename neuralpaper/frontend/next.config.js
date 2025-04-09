/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable CSS modules for Monaco Editor
  webpack: (config, { isServer }) => {
    // Fixes npm packages that depend on `fs` module
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }

    return config;
  },
  reactStrictMode: true,
  images: {
    domains: ['localhost', 'neuralpaper-api.onrender.com'],
  },
  async rewrites() {
    // Get API URL from environment variable or use localhost as default
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8002';
    console.log(`Using API URL: ${apiUrl}`);

    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/:path*`, // Proxy to Backend
      },
    ];
  },
}

module.exports = nextConfig
