import { useEffect, useRef } from "react"

interface BenchwarmerLogoProps {
    className?: string
    /** Compact mode for the header — smaller, fewer particles, no tagline */
    compact?: boolean
}

export function BenchwarmerLogo({ className, compact = false }: BenchwarmerLogoProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const animRef = useRef<number>(0)

    useEffect(() => {
        const canvas = canvasRef.current
        const container = containerRef.current
        if (!canvas || !container) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const resize = () => {
            const rect = container.getBoundingClientRect()
            const dpr = window.devicePixelRatio || 1
            canvas.width = rect.width * dpr
            canvas.height = rect.height * dpr
            canvas.style.width = `${rect.width}px`
            canvas.style.height = `${rect.height}px`
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        }
        resize()
        window.addEventListener("resize", resize)

        const particleCount = compact ? 80 : 360
        const particles = Array.from({ length: particleCount }, (_, i) => {
            const angle = (Math.PI * 2 * i) / particleCount + Math.random() * 0.3
            const radiusX = compact
                ? 0.30 + Math.random() * 0.20
                : 0.32 + Math.random() * 0.18
            const radiusY = compact
                ? 0.20 + Math.random() * 0.25
                : 0.25 + Math.random() * 0.2
            const speed = 0.0006 + Math.random() * 0.0014
            const size = compact
                ? 0.5 + Math.random() * 1.2
                : 1.0 + Math.random() * 2.6
            const opacity = compact
                ? 0.1 + Math.random() * 0.3
                : 0.15 + Math.random() * 0.45
            const colorMix = Math.random()
            return { angle, radiusX, radiusY, speed, size, opacity, colorMix }
        })

        const lerp = (a: number, b: number, t: number) => a + (b - a) * t

        const draw = (time: number) => {
            const rect = container.getBoundingClientRect()
            const W = rect.width
            const H = rect.height
            const cx = W / 2
            const cy = compact ? H * 0.5 : H * 0.42

            ctx.clearRect(0, 0, W, H)

            for (const p of particles) {
                p.angle += p.speed

                const x = cx + Math.cos(p.angle) * (W * p.radiusX)
                const y = cy + Math.sin(p.angle) * (H * p.radiusY)

                const behindText =
                    Math.sin(p.angle) > -0.15 &&
                    Math.sin(p.angle) < 0.15 &&
                    Math.abs(x - cx) < W * 0.28

                const depthFade = Math.sin(p.angle) > 0 ? 0.4 : 1.0
                const alpha = behindText ? p.opacity * 0.1 : p.opacity * depthFade

                const twinkle = 0.7 + 0.3 * Math.sin(time * 0.002 + p.angle * 3)

                const r = Math.round(lerp(167, 245, p.colorMix))
                const g = Math.round(lerp(139, 243, p.colorMix))
                const b = Math.round(lerp(250, 255, p.colorMix))

                ctx.beginPath()
                ctx.arc(x, y, p.size * twinkle, 0, Math.PI * 2)
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha * twinkle})`
                ctx.fill()

                if (!compact && p.size > 1.8) {
                    ctx.beginPath()
                    ctx.arc(x, y, p.size * 2.5, 0, Math.PI * 2)
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha * 0.08})`
                    ctx.fill()
                }
            }

            animRef.current = requestAnimationFrame(draw)
        }

        animRef.current = requestAnimationFrame(draw)

        return () => {
            cancelAnimationFrame(animRef.current)
            window.removeEventListener("resize", resize)
        }
    }, [compact])

    return (
        <div
            ref={containerRef}
            className={`relative ${className || ""}`}
            style={{ aspectRatio: compact ? "5 / 1" : "6 / 1.8" }}
        >
            {/* Particle canvas */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 pointer-events-none"
                style={{ zIndex: 0 }}
            />
            {/* Logo text - centered */}
            <div className="absolute inset-0 flex items-center justify-center" style={{ zIndex: 1 }}>
                <svg
                    viewBox={compact ? "0 0 600 60" : "0 0 600 100"}
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className={compact ? "w-[80%] h-auto" : "w-[85%] h-auto"}
                >
                    <defs>
                        <linearGradient
                            id={compact ? "logo-gradient-compact" : "logo-gradient"}
                            x1="0%"
                            y1="0%"
                            x2="100%"
                            y2="0%"
                        >
                            <stop offset="0%" stopColor="#a78bfa" />
                            <stop offset="45%" stopColor="#c4b5fd" />
                            <stop offset="100%" stopColor="#f5f3ff" />
                        </linearGradient>
                    </defs>
                    <text
                        x="300"
                        y={compact ? "38" : "52"}
                        textAnchor="middle"
                        fontFamily="Inter, system-ui, -apple-system, sans-serif"
                        fontWeight="700"
                        fontSize={compact ? "42" : "58"}
                        letterSpacing="-2"
                        fill={`url(#${compact ? "logo-gradient-compact" : "logo-gradient"})`}
                    >
                        benchwarmer.ai
                    </text>
                    {!compact && (
                        <text
                            x="300"
                            y="78"
                            textAnchor="middle"
                            fontFamily="Inter, system-ui, -apple-system, sans-serif"
                            fontWeight="400"
                            fontSize="13"
                            letterSpacing="4"
                            fill="#6b6b6b"
                        >
                            ADVANCING ACADEMIA LIKE NEVER BEFORE
                        </text>
                    )}
                </svg>
            </div>
        </div>
    )
}
