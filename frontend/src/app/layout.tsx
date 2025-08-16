"use client";

import { ReactNode } from "react";
import { ChakraProvider, defaultSystem } from "@chakra-ui/react";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body className={inter.variable}>
        <ChakraProvider value={defaultSystem}>{children}</ChakraProvider>
      </body>
    </html>
  );
}
