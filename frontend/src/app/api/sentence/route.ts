import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const res = await fetch("http://127.0.0.1:8000/api/sentence");

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: "Backend error", details: errText },
        { status: res.status }
      );
    }

    const data = await res.json();
    /**
     * data: {
     *   sentence: string,
     *   timestamp: number
     * }
     */
    return NextResponse.json(data);
  } catch (err: any) {
    console.error("Flask API error:", err.message);
    return NextResponse.json(
      { error: "Failed to call Flask API" },
      { status: 500 }
    );
  }
}