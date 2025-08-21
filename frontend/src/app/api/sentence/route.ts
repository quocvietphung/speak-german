import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  // Azure OpenAI credentials
  const apiKey = process.env.AZURE_OPENAI_API_KEY!;
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
  const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_NAME!;
  const apiVersion = process.env.AZURE_OPENAI_API_VERSION!;

  const topics = [
    "Reisen und Urlaub",
    "Arbeit und Karriere",
    "Essen und Trinken",
    "Gesundheit und Sport",
    "Familie und Freunde",
    "Technologie und Zukunft",
    "Kunst und Kultur",
    "Natur und Umwelt",
  ];
  const randomTopic = topics[Math.floor(Math.random() * topics.length)];

  const url = `${endpoint}openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

  try {
    // Send a request to Azure OpenAI to generate a simple German sentence
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
              "Du bist ein erfahrener Deutschlehrer. " +
              "Antworte jedes Mal nur mit genau EINEM Beispielsatz. " +
              "Der Satz soll thematisch passen und natürlich klingen.",
          },
          {
            role: "user",
            content: `Bitte gib mir einen Beispielsatz zum Thema: "${randomTopic}".`,
          },
        ],
        max_tokens: 80,
        temperature: 0.8,
      }),
    });

    if (!res.ok) {
      throw new Error(`Azure OpenAI request failed with status ${res.status}`);
    }

    const data = await res.json();

    // Extract the generated sentence or provide a fallback message
    const sentence =
      data?.choices?.[0]?.message?.content?.trim() ||
      "Keine Antwort von Azure OpenAI.";

    return NextResponse.json({ topic: randomTopic, sentence });
  } catch (err: any) {
    // Handle errors and return a 500 response
    console.error("Azure OpenAI API error:", err.message);
    return NextResponse.json(
      { error: "Failed to call Azure OpenAI" },
      { status: 500 }
    );
  }
}