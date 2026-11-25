import Link from "next/link";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-6 text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-mono tracking-widest uppercase mb-4">
        404 — Page Not Found
      </div>
      <h1 className="text-4xl font-extrabold text-white mb-2">
        Chapter Lost in the Smog
      </h1>
      <p className="text-gray-400 max-w-md mb-6 text-sm">
        The story section you are looking for does not exist or has been moved.
      </p>
      <Link
        href="/"
        className="px-6 py-2.5 rounded-full bg-amber-500 text-black font-semibold text-sm hover:bg-amber-400 transition-colors"
      >
        Return to Story
      </Link>
    </main>
  );
}
