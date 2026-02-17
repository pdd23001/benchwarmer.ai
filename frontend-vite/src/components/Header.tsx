import { Link, useLocation } from "react-router-dom"
import { cn } from "@/lib/utils"
import { Bot, BarChart2 } from "lucide-react"


export function Header() {
    const location = useLocation()

    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center">
                <div className="mr-4 flex">
                    <Link to="/" className="mr-6 flex items-center space-x-2">
                        <span className="font-bold flex items-center gap-2">
                            Benchwarmer.AI
                        </span>
                    </Link>
                    <nav className="flex items-center space-x-6 text-sm font-medium">
                        <Link
                            to="/"
                            className={cn(
                                "transition-colors hover:text-foreground/80 flex items-center gap-2",
                                location.pathname === "/" ? "text-foreground" : "text-foreground/60"
                            )}
                        >
                            <Bot className="h-4 w-4" />
                            Chat
                        </Link>
                        <Link
                            to="/benchmarks"
                            className={cn(
                                "transition-colors hover:text-foreground/80 flex items-center gap-2",
                                location.pathname === "/benchmarks" ? "text-foreground" : "text-foreground/60"
                            )}
                        >
                            <BarChart2 className="h-4 w-4" />
                            Benchmarks
                        </Link>
                    </nav>
                </div>
            </div>
        </header>
    )
}
