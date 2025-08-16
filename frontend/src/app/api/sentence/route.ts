import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  // Azure OpenAI credentials
  const apiKey = process.env.AZURE_OPENAI_API_KEY!;
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
  const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_NAME!;
  const apiVersion = process.env.AZURE_OPENAI_API_VERSION!;

  const url = `${endpoint}openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": apiKey,
      },
      body: JSON.stringify({
        messages: [
          {
            role: "system",
            content:
              "Du bist ein Deutschlehrer. Antworte nur mit einem kurzen einfachen deutschen Satz.",
          },
          { role: "user", content: "Gib mir einen Beispielsatz zum Üben." },
        ],
        max_tokens: 64,
        temperature: 0.7,
      }),
    });

    const data = await res.json();

    const sentence =
      data?.choices?.[0]?.message?.content?.trim() ||
      "Keine Antwort von Azure OpenAI.";

    return NextResponse.json({ sentence });
  } catch (err: any) {
    console.error("Azure OpenAI API error:", err.message);
    return NextResponse.json(
      { error: "Failed to call Azure OpenAI" },
      { status: 500 }
    );
  }
}