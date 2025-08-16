'use client';
import { ChakraProvider, defaultSystem } from '@chakra-ui/react';

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="vi">
        <body
            style={{
                minHeight: '100vh',
                background: 'var(--background, #f9fafb)',
                color: 'var(--foreground, #22223b)',
                fontFamily:
                    'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif',
            }}
        >
        <ChakraProvider value={defaultSystem}>
            {children}
        </ChakraProvider>
        </body>
        </html>
    );
}
