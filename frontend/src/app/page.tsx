"use client";

import React, { useState, useRef } from "react";
import { Box, Grid } from "@chakra-ui/react";
import RecordingCard from "@/components/RecordingCard";
import ScoreFeedbackCard from "@/components/ScoreFeedbackCard";

export default function Home() {
  const [targetText, setTargetText] = useState("Klicke auf Next Sentence, um zu starten.");
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [mistakes, setMistakes] = useState<string[]>([]);
  const [tip, setTip] = useState("");
  const [teacherFeedback, setTeacherFeedback] = useState("");
  const [transcript, setTranscript] = useState("");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleRecord = async () => {
    if (!recording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
          setRecording(false);
          setLoading(true);

          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          const formData = new FormData();
          formData.append("audio", audioBlob, "recording.webm");
          formData.append("target_text", targetText);

          try {
            const res = await fetch("/api/evaluate", { method: "POST", body: formData });
            const data = await res.json();
            setScore(data.score ?? null);
            setMistakes(Array.isArray(data.mistakes) ? data.mistakes : []);
            setTranscript(data.transcript ?? "");
            setTip(data.tip ?? "");
            setTeacherFeedback(data.teacherFeedback ?? "");
          } catch (err) {
            console.error("API error", err);
          } finally {
            setLoading(false);
          }
        };

        mediaRecorder.start();
        setRecording(true);
      } catch (err) {
        console.error("Mic access error:", err);
      }
    } else {
      // Stop recording (onstop handler will be triggered)
      mediaRecorderRef.current?.stop();
    }
  };

  const nextSentence = async () => {
    // Reset feedback and score state before fetching a new sentence
    setScore(null);
    setMistakes([]);
    setTranscript("");
    setTip("");
    setTeacherFeedback("");

    try {
      // Request a new practice sentence from the backend
      const res = await fetch("/api/sentence", { method: "POST" });
      const data = await res.json();
      setTargetText(data.sentence || "Keine Antwort");
    } catch (err) {
      console.error("Error fetching sentence", err);
      setTargetText("Fehler beim Abrufen des Satzes.");
    }
  };

  return (
    <Box minH="100vh" p={8} bg="gray.50" display="flex" justifyContent="center">
      <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={8} maxW="6xl" w="100%">
        <RecordingCard
          targetText={targetText}
          recording={recording}
          transcript={transcript}
          onNextSentence={nextSentence}
          onRecord={handleRecord}
        />
        <ScoreFeedbackCard
          loading={loading}
          score={score}
          mistakes={mistakes}
          tip={tip}
          teacherFeedback={teacherFeedback}
        />
      </Grid>
    </Box>
  );
}