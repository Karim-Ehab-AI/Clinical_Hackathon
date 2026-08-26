/**
 * Typed client for the clinical RAG backend.
 * Only endpoint used: POST /api/v1/generation/generate
 */

export const DEFAULT_API_BASE_URL = "http://localhost:3000";
const STORAGE_KEY = "clinical-api-base-url";

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") return DEFAULT_API_BASE_URL;
  return window.localStorage.getItem(STORAGE_KEY)?.trim() || DEFAULT_API_BASE_URL;
}

export function setApiBaseUrl(url: string) {
  if (typeof window === "undefined") return;
  const clean = url.trim().replace(/\/+$/, "");
  if (clean && clean !== DEFAULT_API_BASE_URL) {
    window.localStorage.setItem(STORAGE_KEY, clean);
  } else {
    window.localStorage.removeItem(STORAGE_KEY);
  }
}

export type Citation = {
  chunk_id: string;
  document_id: string;
  source: string;
  recommendation_id: string | null;
  pdf_page: number | null;
  section: string | null;
  source_text: string;
  score: number;
  percentage_score: number;
};

export type GenerationResult = {
  is_in_scope: boolean;
  is_knowledge_sufficient: boolean;
  answer: string | null;
  citations: Citation[];
  refusal_reason: string | null;
  provider: string | null;
  model_name: string | null;
  filtered_chunks_count: number;
};

export type GenerateResponse = {
  query: string;
  result: GenerationResult;
  retrieved_chunks_count: number;
  filtered_chunks_count: number;
};

export class ClinicalApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ClinicalApiError";
  }
}

type ValidationDetail = { loc?: unknown[]; msg?: string; type?: string };

function messageFromPayload(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msgs = (detail as ValidationDetail[]).map((d) => d.msg).filter(Boolean);
      if (msgs.length) return msgs.join(" · ");
    }
  }
  if (status === 422) return "Your question could not be processed. Please rephrase it.";
  if (status >= 500) return "The clinical assistant is temporarily unavailable. Please try again.";
  return `Request failed (${status}).`;
}

/** Calls the backend through the app's own proxy route (avoids CORS / ngrok interstitial). */
export async function generateGuidance(query: string): Promise<GenerateResponse> {
  let response: Response;
  try {
    response = await fetch("/api/backend/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, baseUrl: getApiBaseUrl() }),
    });
  } catch {
    throw new ClinicalApiError(
      "Could not reach the clinical assistant. Check your connection or the backend URL.",
    );
  }

  const payload: unknown = await response.json().catch(() => null);

  if (payload && typeof payload === "object" && "upstreamError" in payload) {
    throw new ClinicalApiError(String((payload as { upstreamError: unknown }).upstreamError));
  }

  if (!response.ok) {
    throw new ClinicalApiError(messageFromPayload(payload, response.status), response.status);
  }
  if (!payload || typeof payload !== "object" || !("result" in payload)) {
    throw new ClinicalApiError("The clinical assistant returned an unexpected response.");
  }
  return payload as GenerateResponse;
}
