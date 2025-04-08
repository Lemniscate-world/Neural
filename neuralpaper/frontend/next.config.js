/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost', 'neuralpaper-api.onrender.com'],
  },
  async rewrites() {
    // Get API URL from environment variable or use localhost as default
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
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
