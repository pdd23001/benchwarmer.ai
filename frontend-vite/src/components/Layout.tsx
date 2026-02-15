import { Sidebar } from "@/components/Sidebar"

interface LayoutProps {
    children: React.ReactNode
    hasChat?: boolean
    currentSessionId?: string | null
    codedAlgorithms?: string[]
    onAlgorithmClick?: (name: string) => void
    activeAlgorithm?: string | null
    onSessionClick?: (sessionId: string) => void
    onNewChat?: () => void
}

export function Layout({
    children,
    hasChat = false,
    currentSessionId,
    codedAlgorithms = [],
    onAlgorithmClick,
    activeAlgorithm,
    onSessionClick,
    onNewChat,
}: LayoutProps) {
    return (
        <div className="flex h-screen bg-background overflow-hidden">
            <Sidebar
                hasChat={hasChat}
                currentSessionId={currentSessionId}
                codedAlgorithms={codedAlgorithms}
                onAlgorithmClick={onAlgorithmClick}
                activeAlgorithm={activeAlgorithm}
                onSessionClick={onSessionClick}
                onNewChat={onNewChat}
            />
            <main className="flex-1 flex flex-col relative w-full overflow-hidden">
                {children}
            </main>
        </div>
    )
}
