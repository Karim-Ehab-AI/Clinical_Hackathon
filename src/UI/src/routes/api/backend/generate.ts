import { createFileRoute } from "@tanstack/react-router";

const DEFAULT_BASE_URL = "http://localhost:3000";

type ProxyBody = { query?: unknown; baseUrl?: unknown };

export const Route = createFileRoute("/api/backend/generate")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        let body: ProxyBody;
        try {
          body = (await request.json()) as ProxyBody;
        } catch {
          return Response.json({ detail: "Invalid request body." }, { status: 400 });
        }

        const query = typeof body.query === "string" ? body.query.trim() : "";
        if (query.length < 2) {
          return Response.json(
            { detail: "Please describe the situation in at least 2 characters." },
            { status: 422 },
          );
        }

        const rawBase = typeof body.baseUrl === "string" && body.baseUrl.trim()
          ? body.baseUrl.trim()
          : DEFAULT_BASE_URL;
        const base = rawBase.replace(/\/+$/, "");

        let upstream: Response;
        try {
          upstream = await fetch(`${base}/api/v1/generation/generate`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
              "ngrok-skip-browser-warning": "true",
            },
            body: JSON.stringify({ query }),
          });
        } catch {
          // Return 200 with an error payload: an upstream outage is not an app
          // failure, and a 5xx here would trip the preview error overlay.
          return Response.json({
            upstreamError:
              "Could not reach the clinical backend. Make sure the backend URL is running and correct.",
          });
        }

        const text = await upstream.text();
        const contentType = upstream.headers.get("content-type") ?? "";

        if (!contentType.includes("application/json")) {
          return Response.json({
            upstreamError: `The backend URL did not return a clinical API response (status ${upstream.status}). Check the base URL.`,
          });
        }

        if (upstream.status >= 500) {
          return Response.json({
            upstreamError:
              "The clinical assistant is temporarily unavailable. Please try again.",
          });
        }

        return new Response(text, {
          status: upstream.status,
          headers: { "Content-Type": "application/json" },
        });
      },
    },
  },
});
