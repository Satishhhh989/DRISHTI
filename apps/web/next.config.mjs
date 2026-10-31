/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    "@photo-to-3d/scene-schema",
    "@photo-to-3d/shared-types"
  ],
  reactStrictMode: true,
};

export default nextConfig;

