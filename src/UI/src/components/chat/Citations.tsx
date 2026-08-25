import { BookMarked, ChevronDown } from "lucide-react";
import { useState } from "react";

import type { Citation } from "@/lib/clinical-api";
import { cn } from "@/lib/utils";
import { isRtl } from "./types";

export function Citations({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (!citations.length) return null;

  return (
    <div className="mt-3 rounded-xl border border-border bg-secondary/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-secondary-foreground"
      >
        <BookMarked className="size-3.5 text-primary" />
        {citations.length} clinical {citations.length === 1 ? "source" : "sources"}
        <ChevronDown
          className={cn("ml-auto size-3.5 transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <ul className="space-y-2 border-t border-border px-3 py-3">
          {citations.map((c) => (
            <li key={c.chunk_id} className="rounded-lg bg-background p-3 text-xs">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-muted-foreground">
                {c.section && <span className="font-medium text-foreground">{c.section}</span>}
                {c.recommendation_id && <span>rec. {c.recommendation_id}</span>}
                <span className="ml-auto rounded-full bg-accent px-2 py-0.5 font-medium text-accent-foreground">
                  {Math.round(c.percentage_score)}% match
                </span>
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground/80">
                <span className="truncate max-w-[16rem]" title={c.source}>
                  Source: {c.source}
                </span>
                {c.pdf_page != null && <span>Page {c.pdf_page}</span>}
                <span>Chunk {c.chunk_id}</span>
              </div>
              <p
                dir={isRtl(c.source_text) ? "rtl" : "ltr"}
                className="mt-2 leading-relaxed text-muted-foreground"
              >
                {c.source_text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
