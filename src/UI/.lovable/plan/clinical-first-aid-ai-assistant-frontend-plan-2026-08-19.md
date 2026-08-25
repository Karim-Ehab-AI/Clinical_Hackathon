# Clinical First-Aid AI Assistant — Frontend Plan

## What I understood

A desktop-first, ChatGPT-style chat UI over your existing clinical RAG backend. Only one endpoint is needed: `POST /api/v1/generation/generate` with `{ "query": "..." }` (min 2 chars). No upload, no auth, no dashboards, no extra pages.

Responses are handled per your five scenarios:
- Answer + citations -> render the answer (bullets/steps/warnings) and a compact collapsible "Sources" list showing section, PDF page, recommendation id, and match percentage.
- `is_knowledge_sufficient: false` or `is_in_scope: false` -> render `refusal_reason` as a calm, clearly-styled notice bubble (not an error).
- 422 -> inline validation message from `detail[].msg`.
- 500 / network / ngrok offline -> friendly "assistant is unreachable" message with a Retry action on that message.

Answers may come back in Arabic, so message bubbles auto-detect direction and render RTL when needed.

## What I'll build

Single page at `/`:
- Compact top bar: small brand mark (generated clinical logo) + product name, plus a small settings button for the backend URL.
- Empty state centered: welcome title, one-line instruction, and 3–4 example prompts that fill the input.
- Conversation view: user bubbles (blue, high-contrast) vs assistant messages (plain on white with a small clinical avatar), markdown-rendered, subtle fade/slide-in animation.
- Sticky bottom composer: single textarea, Enter to send, Shift+Enter newline, send button, disabled while loading.
- Typing indicator: shimmer "Analyzing clinical guidance…" while the request is in flight.
- Session-only history (in-memory), plus a "New conversation" button.
- One-time medical disclaimer line under the composer (small, muted).

Visual system: white dominant, blue/cyan accent tokens only, rounded corners, soft shadows, modern sans typography. All colors as semantic tokens in `src/styles.css`.

## Backend integration

- Default base URL: `https://undercook-cogwheel-undone.ngrok-free.dev`, overridable at runtime from the settings popover (persisted in localStorage) so you can swap the ngrok URL without a rebuild.
- Calls go through a thin server proxy route (`/api/backend/generate`) so browser CORS and ngrok's browser-warning interstitial can't break the app; the proxy forwards the JSON body and adds `ngrok-skip-browser-warning`. The active base URL travels with the request so your override still applies.
- No timeouts that cut off slow Colab responses; a per-message Retry handles failures.

## Technical notes

- TanStack Start; `src/routes/index.tsx` is the chat page, `src/routes/api/backend/generate.ts` the proxy.
- `src/lib/clinical-api.ts`: typed `GenerateResponse` (result: is_in_scope, is_knowledge_sufficient, answer, citations[], refusal_reason, provider, model_name, filtered_chunks_count) + error normalization.
- `src/lib/api-config.ts`: base-URL read/write helper.
- Components: `ChatMessage`, `Citations`, `Composer`, `EmptyState`, `BrandBar` under `src/components/chat/`.
- `react-markdown` for assistant formatting; AI Elements primitives for conversation/message/prompt-input/shimmer.
- Route `head()` with clinical-specific title/description/OG tags.

## Out of scope

Ingestion/upload UI, retrieval/search page, auth, persistence across reloads.
