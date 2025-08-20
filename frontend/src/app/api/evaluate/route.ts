import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const audio = formData.get("audio") as File | null;
    const targetText = formData.get("target_text") as string | null;

    if (!audio || !targetText) {
      return NextResponse.json(
        { error: "Missing audio or target_text" },
        { status: 400 }
      );
    }

    const bytes = await audio.arrayBuffer();
    const blob = new Blob([bytes], { type: audio.type });

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
        { error: "Backend error", details: errText },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();

    // Trả nguyên kết quả từ Flask
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      { error: "Server error", details: err.message },
      { status: 500 }
    );
  }
}