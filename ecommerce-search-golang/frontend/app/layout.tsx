import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { AppSidebar } from '@/components/app-sidebar'

import '@/app/globals.css'
import { SearchProvider } from '@/contexts/search-context'
import { QueryProvider } from '@/components/providers/query-provider'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'Qdrant E-commerce Demo',
  description:
    'A demo e-commerce application using Qdrant as a vector database.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-zinc-50 font-sans`}
      >
        <QueryProvider>
          <SearchProvider>
            <SidebarProvider>
              <AppSidebar />
              <main className="w-full">
                <div className="h-8 p-2">
                  <SidebarTrigger />
                </div>
                {children}
              </main>
            </SidebarProvider>
          </SearchProvider>
        </QueryProvider>
      </body>
    </html>
  )
}
