import logo from "@/assets/logo.png";

const EXAMPLES = [
  "Someone has a second-degree burn on their forearm — what do I do?",
  "An adult is choking and cannot speak. Guide me step by step.",
  "How do I control severe bleeding from a deep cut?",
  "A person collapsed and is unresponsive but breathing.",
];

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="animate-rise mx-auto flex max-w-2xl flex-col items-center px-6 py-16 text-center">
      <img src={logo} alt="" width={48} height={48} className="size-12" />
      <h1 className="mt-5 text-2xl font-semibold tracking-tight text-foreground">
        Describe the first-aid situation
      </h1>
      <p className="mt-2 max-w-lg text-sm leading-6 text-muted-foreground">
        Tell the assistant what happened in plain words. It answers only from verified clinical
        first-aid guidelines, with the exact sources it used.
      </p>

      <div className="mt-8 grid w-full gap-2 sm:grid-cols-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onPick(example)}
            className="rounded-xl border border-border bg-card p-3.5 text-start text-sm leading-6 text-secondary-foreground transition-colors hover:border-primary/40 hover:bg-accent/50"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
