import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PHOTO → 3D ENGINE",
  description: "Local-first spatial reconstruction engine transforming multi-person photographs into interactive 3D scenes.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
