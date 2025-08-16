import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

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

  const result = await response.json();
  return NextResponse.json(result);
}