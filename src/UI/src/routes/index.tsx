import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { BrandBar } from "@/components/chat/BrandBar";
import { EmptyState } from "@/components/chat/EmptyState";
import { MessageBubble } from "@/components/chat/MessageBubble";
import type { ChatMessage } from "@/components/chat/types";
import { ClinicalApiError, generateGuidance } from "@/lib/clinical-api";

const TITLE = "MedAid Clinical Assistant — AI First-Aid Guidance";
const DESCRIPTION =
  "Describe a first-aid or emergency situation and get step-by-step clinical guidance sourced from verified medical guidelines.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ChatPage,
});

function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const idRef = useRef(0);
  const nextId = () => `m${++idRef.current}`;

  const ask = useCallback(async (query: string, options?: { skipUserMessage?: boolean }) => {
    const trimmed = query.trim();
    if (trimmed.length < 2 || isLoading) return;

    setInput("");
    setIsLoading(true);
    if (!options?.skipUserMessage) {
      setMessages((prev) => [...prev, { id: nextId(), role: "user", text: trimmed }]);
    }

    try {
      const data = await generateGuidance(trimmed);
      const { result } = data;

      setMessages((prev) => [
        ...prev,
        result.answer
          ? {
              id: nextId(),
              role: "assistant",
              kind: "answer",
              text: result.answer,
              citations: result.citations ?? [],
              modelName: result.model_name ?? null,
            }
          : {
              id: nextId(),
              role: "assistant",
              kind: "refusal",
              text:
                result.refusal_reason ??
                "No reliable first-aid guidance could be found for this question.",
            },
      ]);
    } catch (error) {
      const text =
        error instanceof ClinicalApiError
          ? error.message
          : "Unexpected error while contacting the clinical assistant.";
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", kind: "error", text, retryQuery: trimmed },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const retry = useCallback(
    (query: string) => {
      setMessages((prev) => prev.filter((m) => !(m.role === "assistant" && m.kind === "error")));
      void ask(query, { skipUserMessage: true });
    },
    [ask],
  );

  const hasConversation = messages.length > 0;

  return (
    <div className="flex h-screen flex-col bg-background">
      <BrandBar
        canReset={hasConversation && !isLoading}
        onReset={() => {
          setMessages([]);
          setInput("");
        }}
      />

      <main className="flex min-h-0 flex-1 flex-col">
        {hasConversation ? (
          <Conversation className="flex-1">
            <ConversationContent className="mx-auto w-full max-w-3xl space-y-6 px-6 py-8">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} onRetry={retry} />
              ))}
              {isLoading && (
                <div className="flex items-center gap-3 ps-10">
                  <Shimmer className="text-sm">Reviewing clinical guidelines…</Shimmer>
                </div>
              )}
            </ConversationContent>
            <ConversationScrollButton />
          </Conversation>
        ) : (
          <div className="flex flex-1 items-center justify-center overflow-y-auto">
            <EmptyState onPick={(text) => setInput(text)} />
          </div>
        )}

        <div className="border-t border-border bg-background/90 backdrop-blur">
          <div className="mx-auto w-full max-w-3xl px-6 py-4">
            <PromptInput
              className="rounded-2xl shadow-[var(--shadow-composer)]"
              onSubmit={(message, event) => {
                event.preventDefault();
                void ask(message.text ?? input);
              }}
            >
              <PromptInputTextarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Describe the situation — e.g. “A child swallowed a small object and is coughing”"
                disabled={isLoading}
                className="min-h-[64px]"
              />
              <PromptInputFooter className="justify-end">
                <PromptInputSubmit
                  {...(isLoading ? { status: "submitted" as const } : {})}
                  disabled={isLoading || input.trim().length < 2}
                />
              </PromptInputFooter>
            </PromptInput>
            <p className="mt-2 text-center text-[11px] leading-5 text-muted-foreground">
              Educational first-aid support only. In a life-threatening emergency, call your local
              emergency number immediately.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
