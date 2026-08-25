import type { Citation } from "@/lib/clinical-api";

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      kind: "answer";
      text: string;
      citations: Citation[];
      modelName: string | null;
    }
  | { id: string; role: "assistant"; kind: "refusal"; text: string }
  | { id: string; role: "assistant"; kind: "error"; text: string; retryQuery: string };

export function isRtl(text: string) {
  return /[\u0600-\u06FF\u0750-\u077F]/.test(text);
}
