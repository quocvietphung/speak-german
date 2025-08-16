// app/api/evaluate/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("audio") as File;
  const targetText = formData.get("target_text") as string;

  if (!file) {
    return NextResponse.json({ error: "No audio provided" }, { status: 400 });
  }

  // Gọi Hugging Face Whisper API
  const response = await fetch(
    "https://api-inference.huggingface.co/models/openai/whisper-small",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HF_API_TOKEN}`,
      },
      body: file,
    }
  );

  const whisperResult = await response.json();
  const transcript = whisperResult.text || "";

  // So sánh transcript với targetText → tính score, mistakes
  const targetWords = targetText.toLowerCase().split(/\s+/);
  const spokenWords = transcript.toLowerCase().split(/\s+/);

  const mistakes = targetWords.filter((w) => !spokenWords.includes(w));
  const correct = targetWords.length - mistakes.length;
  const score = Math.max(0, Math.round((correct / targetWords.length) * 100));

  const tip =
    mistakes.length > 0
      ? `Try to pronounce: ${mistakes.join(", ")}`
      : "Perfect! Keep it up 🎉";

  return NextResponse.json({
    transcript,
    score,
    mistakes,
    tip,
  });
}