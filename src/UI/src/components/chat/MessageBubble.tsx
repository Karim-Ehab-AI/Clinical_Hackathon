import { AlertTriangle, RotateCcw, ShieldAlert, Stethoscope } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Citations } from "./Citations";
import { isRtl, type ChatMessage } from "./types";

function AssistantAvatar() {
  return (
    <div
      className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg text-primary-foreground"
      style={{ backgroundImage: "var(--gradient-clinical)" }}
      aria-hidden
    >
      <Stethoscope className="size-4" />
    </div>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <div
      dir={isRtl(text) ? "rtl" : "ltr"}
      className={cn(
        "text-[0.9375rem] leading-7 text-foreground",
        "[&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-1",
        "[&_ul]:list-disc [&_ol]:list-decimal [&_ul]:ps-5 [&_ol]:ps-5",
        "[&_strong]:font-semibold [&_strong]:text-foreground",
        "[&_h1]:mt-4 [&_h1]:mb-2 [&_h1]:text-base [&_h1]:font-semibold",
        "[&_h2]:mt-4 [&_h2]:mb-2 [&_h2]:text-base [&_h2]:font-semibold",
        "[&_h3]:mt-3 [&_h3]:mb-1.5 [&_h3]:font-semibold",
        "[&_blockquote]:border-s-2 [&_blockquote]:border-primary [&_blockquote]:ps-3 [&_blockquote]:text-muted-foreground",
        "[&_code]:rounded [&_code]:bg-secondary [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[0.85em]",
        "[&_a]:text-primary [&_a]:underline [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

export function MessageBubble({
  message,
  onRetry,
}: {
  message: ChatMessage;
  onRetry: (query: string) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="animate-rise flex w-full justify-end">
        <div
          dir={isRtl(message.text) ? "rtl" : "ltr"}
          className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[0.9375rem] leading-6 text-primary-foreground shadow-[var(--shadow-soft)]"
        >
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-rise flex w-full gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1 pt-0.5">
        {message.kind === "answer" && (
          <>
            <Markdown text={message.text} />
            <Citations citations={message.citations} />
          </>
        )}

        {message.kind === "refusal" && (
          <div className="rounded-xl border border-border bg-accent/60 p-4">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-accent-foreground">
              <ShieldAlert className="size-3.5" />
              Safety guardrail
            </div>
            <p
              dir={isRtl(message.text) ? "rtl" : "ltr"}
              className="text-[0.9375rem] leading-7 text-foreground"
            >
              {message.text}
            </p>
          </div>
        )}

        {message.kind === "error" && (
          <div className="rounded-xl border border-destructive/25 bg-destructive/5 p-4">
            <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-destructive">
              <AlertTriangle className="size-3.5" />
              Something went wrong
            </div>
            <p className="text-[0.9375rem] leading-6 text-foreground">{message.text}</p>
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={() => onRetry(message.retryQuery)}
            >
              <RotateCcw className="size-3.5" />
              Retry
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
