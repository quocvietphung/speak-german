import {NextRequest, NextResponse} from "next/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData();
        const audio = formData.get("audio") as File | null;
        const targetText = formData.get("target_text") as string | null;

        if (!audio || !targetText) {
            return NextResponse.json(
                {error: "Missing audio or target_text"},
                {status: 400}
            );
        }

        // 🔄 Convert File -> Blob (to avoid errors when forwarding in Node.js)
        const bytes = await audio.arrayBuffer();
        const blob = new Blob([bytes], {type: audio.type});

        // Send audio file + target sentence to FastAPI backend
        const forwardData = new FormData();
        forwardData.append("audio", blob, "recording.webm");
        forwardData.append("target_text", targetText);

        const backendRes = await fetch("http://127.0.0.1:8000/api/evaluate", {
            method: "POST",
            body: forwardData,
        });

        if (!backendRes.ok) {
            const errText = await backendRes.text();
            return NextResponse.json(
                {error: "Backend error", details: errText},
                {status: backendRes.status}
            );
        }

        const data = await backendRes.json();
        /**
         * data: {
         *   reference: string,
         *   hypothesis: string,   // <- this is the transcript
         *   score: number,
         *   mistakes: string[],
         *   tip: string
         * }
         */

        // Call Azure OpenAI to generate deeper feedback
        const apiKey = process.env.AZURE_OPENAI_API_KEY!;
        const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
        const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_NAME!;
        const apiVersion = process.env.AZURE_OPENAI_API_VERSION!;
        const url = `${endpoint}openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

        let teacherFeedback = "";
        try {
            const aiRes = await fetch(url, {
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
                                "Du bist ein freundlicher deutscher Sprachlehrer mit phonetik-Fokus. Antworte klar, motivierend, maximal 3–4 Sätze. Gib für jedes Fehlerwort:\n- korrekte Version,\n- IPA (inkl. Schwa- oder Zentralvokal-Spielarten),\n- Betonung (Hauptakzent),\n- ob Vokal lang/kurz,\n- Beispiel zur Wiederholung.",
                        },
                        {
                            role: "user",
                            content: `Reference sentence: ${data.reference}
                                      Student said: ${data.hypothesis}
                                      Score: ${data.score}%
                                      Mistake words: ${data.mistakes?.join(", ") || "none"}
                                      Technical tip: ${data.tip}
                                      
                                      Please provide teacher feedback including pronunciation analysis as described above.`,
                        },
                    ],
                    max_tokens: 256,
                    temperature: 0.7,
                }),
            });

            if (aiRes.ok) {
                const aiData = await aiRes.json();
                teacherFeedback =
                    aiData?.choices?.[0]?.message?.content?.trim() ||
                    data.tip ||
                    "Good job!";
            } else {
                const errText = await aiRes.text();
                console.error("Azure OpenAI error:", errText);
                teacherFeedback = data.tip || "Good job!";
            }
        } catch (err) {
            console.error("Azure OpenAI error:", err);
            teacherFeedback = data.tip || "Good job!";
        }

        return NextResponse.json({
            ...data,
            transcript: data.hypothesis,
            teacherFeedback,
        });
    } catch (err: any) {
        return NextResponse.json(
            {error: "Server error", details: err.message},
            {status: 500}
        );
    }
}